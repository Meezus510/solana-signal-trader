"""
trader/trading/engine.py — Multi-strategy coordinator.

MultiStrategyEngine wires together N StrategyRunners and a shared Birdeye
price feed. Its only responsibilities are:

    1. handle_new_signal()   — fetch entry price once, fan buy to all runners
    2. monitor_positions()   — collect unique mints across all runners,
                               fetch prices once, distribute to each runner
    3. print_summary()       — per-strategy summaries + global aggregate

All strategy evaluation logic lives in StrategyRunner (strategy.py) so
this class stays free of trading rules and easy to extend.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from trader.analysis.chart import OHLCV_BARS, compute_chart_context
from trader.analysis.ml_scorer import (
    ChartMLScorer, ML_OHLCV_BARS, ML_OHLCV_INTERVAL,
    MORALIS_OHLCV_BARS, MORALIS_OHLCV_INTERVAL,
)
from trader.config import Config
from trader.pricing.birdeye import BirdeyePriceClient
from trader.pricing.moralis import MoralisOHLCVClient
from trader.trading.models import TokenSignal
from trader.trading.strategy import StrategyRunner

_STRATEGY_CONFIG_PATH = Path(__file__).parent.parent.parent / "strategy_config.json"

logger = logging.getLogger(__name__)
signal_log = logging.getLogger("signals")


class MultiStrategyEngine:
    """
    Coordinates N StrategyRunners with a single shared Birdeye price feed.

    Shared price polling:
        Each poll cycle collects every unique mint across all runners'
        open positions, fetches prices in one batch call, then fans each
        price out to every runner that holds that mint. A mint held by
        3 strategies still costs exactly 1 Birdeye API call per cycle.

    Independent exit logic:
        Each runner evaluates its own stop-loss / take-profit / trailing
        stop independently. One strategy closing a position does not
        affect any other strategy holding the same mint.
    """

    def __init__(
        self,
        cfg: Config,
        runners: list[StrategyRunner],
        birdeye_client: BirdeyePriceClient,
        db=None,
        moralis_client: MoralisOHLCVClient | None = None,
    ) -> None:
        self._cfg = cfg
        self._runners = runners
        self._birdeye = birdeye_client
        self._moralis = moralis_client
        self._db = db
        # Mints currently awaiting reanalysis — prevents duplicate scheduling
        self._pending_reanalysis: set[str] = set()
        # "(mint:runner_name)" keys awaiting AI override re-check
        self._pending_ai_override: set[str] = set()
        # One ML scorer per unique training strategy, keyed by training strategy name.
        # Chart variants point to their base strategy so they train on unbiased data.
        self._ml_scorers: dict[str, ChartMLScorer] = {}
        # Strategies that prefer Moralis 10s candles for KNN scoring (fast-scalp only).
        # All others fall back to Birdeye 1m so features match their longer-horizon data.
        self._ml_prefer_moralis: set[str] = set()
        if db:
            for r in runners:
                if r.cfg.save_chart_data:  # score whenever chart data is saved, even if filter is off
                    training = r.cfg.ml_training_strategy or r.cfg.name
                    scorer_key = f"{training}::{r.cfg.ml_training_label}"
                    if scorer_key not in self._ml_scorers:
                        self._ml_scorers[scorer_key] = ChartMLScorer(
                            db, strategy=training,
                            k=r.cfg.ml_k,
                            recency_halflife_days=r.cfg.ml_halflife_days,
                            score_low_pct=r.cfg.ml_score_low_pct,
                            score_high_pct=r.cfg.ml_score_high_pct,
                            training_label=r.cfg.ml_training_label,
                            feature_weights=list(r.cfg.ml_feature_weights)
                                if r.cfg.ml_feature_weights else None,
                        )
                    if r.cfg.ml_prefer_moralis:
                        self._ml_prefer_moralis.add(scorer_key)
        logger.info(
            "[engine] Initialized — %d runner(s) | ml_scorers: %s | moralis_knn: %s",
            len(self._runners),
            list(self._ml_scorers.keys()) or "none",
            list(self._ml_prefer_moralis) or "none",
        )
        # Track strategy_config.json mtime for hot-reload
        self._config_mtime: float = self._get_config_mtime()

    # ------------------------------------------------------------------
    # Hot-reload
    # ------------------------------------------------------------------

    @staticmethod
    def _get_config_mtime() -> float:
        try:
            return _STRATEGY_CONFIG_PATH.stat().st_mtime
        except OSError:
            return 0.0

    def _check_config_reload(self) -> None:
        """
        Reload runners if strategy_config.json has been modified since last check.

        Open positions and cash balances are migrated to the new runners so
        in-flight trades are unaffected. New parameters apply to the next signal.
        """
        mtime = self._get_config_mtime()
        if mtime <= self._config_mtime:
            return

        logger.info("[engine] strategy_config.json changed — hot-reloading runners")
        self._config_mtime = mtime

        try:
            from trader.strategies.registry import build_runners
            new_runners = build_runners(self._cfg, self._db)
        except Exception:
            logger.exception("[engine] Hot-reload failed — keeping existing runners")
            return

        # Migrate state from old runners to new runners by name
        old_by_name = {r.name: r for r in self._runners}
        for new_runner in new_runners:
            old = old_by_name.get(new_runner.name)
            if old is None:
                continue
            # Restore cash balance
            ps = old.portfolio_state
            new_runner.restore_cash(ps.available_cash_usd, ps.starting_cash_usd)
            # Restore open positions (they keep their original entry/stop/TP prices)
            open_positions = old.get_open_positions()
            if open_positions:
                new_runner.restore_positions(open_positions)
            # Restore pending outcome IDs for open positions
            new_runner._outcome_ids = dict(old._outcome_ids)

        self._runners = new_runners

        # Rebuild ML scorers to reflect any use_ml_filter or hyperparameter changes
        self._ml_scorers = {}
        self._ml_prefer_moralis = set()
        if self._db:
            for r in self._runners:
                if r.cfg.save_chart_data:  # score whenever chart data is saved, even if filter is off
                    training = r.cfg.ml_training_strategy or r.cfg.name
                    scorer_key = f"{training}::{r.cfg.ml_training_label}"
                    if scorer_key not in self._ml_scorers:
                        self._ml_scorers[scorer_key] = ChartMLScorer(
                            self._db, strategy=training,
                            k=r.cfg.ml_k,
                            recency_halflife_days=r.cfg.ml_halflife_days,
                            score_low_pct=r.cfg.ml_score_low_pct,
                            score_high_pct=r.cfg.ml_score_high_pct,
                            training_label=r.cfg.ml_training_label,
                            feature_weights=list(r.cfg.ml_feature_weights)
                                if r.cfg.ml_feature_weights else None,
                        )
                    if r.cfg.ml_prefer_moralis:
                        self._ml_prefer_moralis.add(scorer_key)

        logger.info(
            "[engine] Hot-reload complete — %d runner(s) active | ml_scorers: %s",
            len(self._runners),
            list(self._ml_scorers.keys()) or "none",
        )

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    async def handle_new_signal(self, signal: TokenSignal) -> None:
        """
        Process a new token signal across all strategies.

        Entry price is fetched once and shared — N strategies entering
        the same token costs 1 Birdeye call, not N calls.
        """
        self._check_config_reload()

        logger.info("[SIGNAL] %s | mint=%s | channel=%s", signal.symbol, signal.mint_address, signal.source_channel)
        signal_log.info("SIGNAL     | %-10s | %-44s | ch=%-20s", signal.symbol, signal.mint_address, signal.source_channel)
        if self._db:
            self._db.log_signal("SIGNAL", symbol=signal.symbol, mint=signal.mint_address, source_channel=signal.source_channel)

        if self._cfg.dry_run:
            logger.info("[DRY_RUN] %s — data collection mode (no positions)", signal.symbol)

        entry_price = await self._birdeye.get_price(signal.mint_address)
        if entry_price is None:
            logger.warning("[SKIP] No live price for %s — entry aborted", signal.symbol)
            signal_log.info("NO_PRICE   | %-10s | %-44s | ch=%-20s", signal.symbol, signal.mint_address, signal.source_channel)
            if self._db:
                self._db.log_signal("NO_PRICE", symbol=signal.symbol, mint=signal.mint_address, source_channel=signal.source_channel)
            return

        logger.info(
            "[ENTRY] %s @ $%.8f — distributing to %d strategy(s)",
            signal.symbol, entry_price, len(self._runners),
        )

        # Fetch 1m Birdeye candles if any runner uses the chart filter OR saves
        # chart data (stored as candles_1m_json for future strategy use).
        candles = []
        chart_ctx = None
        needs_1m = any(r.cfg.use_chart_filter or r.cfg.save_chart_data for r in self._runners)
        if needs_1m:
            candles = await self._birdeye.get_ohlcv(signal.mint_address, bars=OHLCV_BARS)
            chart_ctx = compute_chart_context(candles, entry_price)
            if chart_ctx:
                logger.info(
                    "[CHART] %s | pump=%.1fx | vol=%s | enter=%s | %s",
                    signal.symbol, chart_ctx.pump_ratio, chart_ctx.vol_trend,
                    chart_ctx.should_enter, chart_ctx.reason,
                )
            else:
                logger.info(
                    "[CHART] %s | insufficient candle data — chart filter bypassed",
                    signal.symbol,
                )

        # ------------------------------------------------------------------
        # ML candles — try Moralis (high-res) first, fall back to Birdeye.
        # Any Moralis failure is caught silently; Birdeye is always the safety net.
        # The interval/bars are controlled by MORALIS_OHLCV_INTERVAL /
        # MORALIS_OHLCV_BARS in ml_scorer.py — change those constants to switch
        # to any supported resolution (10s, 30s, 1min, etc.).
        # ------------------------------------------------------------------
        # Fetch ML candles for any runner that scores OR saves chart data.
        # Two separate sources are tracked for the dual-resolution feature vector:
        #   moralis_candles — Moralis 10s (features 1-6, empty if Moralis fails)
        #   candles         — Birdeye 1m (features 7-12, already fetched above)
        # ml_candles is the primary candle list stored in signal_charts.candles_json:
        #   Moralis 10s when available, else Birdeye 1m as storage fallback.
        # ------------------------------------------------------------------
        needs_ml_candles = self._ml_scorers or any(r.cfg.save_chart_data for r in self._runners)
        moralis_candles = []   # raw Moralis 10s (may be empty — used for features 1-6)
        ml_candles      = []   # primary storage candles (Moralis 10s or Birdeye 1m fallback)
        ml_source       = "none"
        if needs_ml_candles:
            if self._moralis:
                try:
                    moralis_candles = await self._moralis.get_ohlcv(
                        signal.mint_address,
                        bars=MORALIS_OHLCV_BARS,
                        interval=MORALIS_OHLCV_INTERVAL,
                    )
                    if moralis_candles:
                        ml_candles = moralis_candles
                        ml_source  = f"moralis/{MORALIS_OHLCV_INTERVAL}"
                    else:
                        logger.warning(
                            "[ML] Moralis returned no candles for %s — 10s features will be neutral",
                            signal.symbol,
                        )
                except Exception as exc:
                    logger.warning(
                        "[ML] Moralis OHLCV error for %s — 10s features will be neutral: %s",
                        signal.symbol, exc,
                    )

            if not ml_candles:
                # Storage fallback only — Birdeye 1m is also used as features 7-12,
                # so no separate fetch is needed; `candles` already has 1m data.
                if candles:
                    ml_candles = candles
                    ml_source  = "birdeye/1m (fallback)"
                else:
                    logger.warning("[ML] No candles available for %s", signal.symbol)

        # Moralis pair stats — completely optional, never blocks scoring.
        pair_stats = None
        if needs_ml_candles and self._moralis:
            try:
                pair_stats = await self._moralis.get_pair_stats(signal.mint_address)
                if pair_stats is None:
                    logger.warning("[ML] Moralis pair stats returned nothing for %s", signal.symbol)
            except Exception as exc:
                logger.warning("[ML] Moralis pair stats error for %s: %s", signal.symbol, exc)

        # Birdeye token overview — market cap, absolute liquidity, holder count.
        # Merged into pair_stats so both are persisted together in pair_stats_json
        # and flow through to ML features automatically.  Completely optional.
        if needs_ml_candles:
            try:
                token_meta = await self._birdeye.get_token_overview(signal.mint_address)
                if token_meta:
                    pair_stats = {**(pair_stats or {}), **token_meta}
                    logger.debug(
                        "[TokenOverview] %s | mc=%.0f liq=%.0f holders=%s",
                        signal.symbol,
                        token_meta.get("market_cap_usd") or 0,
                        token_meta.get("liquidity_usd") or 0,
                        token_meta.get("holder_count"),
                    )
            except Exception as exc:
                logger.warning("[TokenOverview] error for %s: %s", signal.symbol, exc)

        # ML confidence score — one score per unique training strategy.
        # Chart variants train on their base strategy's unfiltered outcomes.
        #
        # Dual-resolution scoring:
        #   ml_prefer_moralis=True  (quick_pop) → candles_10s=moralis_candles (may be []),
        #                                          candles_1m=candles (Birdeye 1m, always).
        #   ml_prefer_moralis=False (trend/moonbag) → candles_10s=[], candles_1m=candles.
        ml_scores: dict[str, float | None] = {}
        if self._ml_scorers and (moralis_candles or candles):
            for scorer_key, scorer in self._ml_scorers.items():
                use_moralis = scorer_key in self._ml_prefer_moralis
                score_10s   = moralis_candles if use_moralis else []
                score_1m    = candles or None
                score = scorer.score(
                    score_10s,
                    candles_1m=score_1m,
                    chart_ctx=chart_ctx,
                    pair_stats=pair_stats,
                    source_channel=signal.source_channel,
                )
                ml_scores[scorer_key] = score
                if score is not None:
                    knn_src = (
                        f"dual(moralis/{MORALIS_OHLCV_INTERVAL}+birdeye/1m)"
                        if use_moralis else "birdeye/1m"
                    )
                    training_name = scorer_key.split("::")[0]
                    logger.info(
                        "[ML] %s | confidence=%.1f/10 | knn_src=%s | pair_stats=%s | training=%s",
                        signal.symbol, score, knn_src, "yes" if pair_stats else "no",
                        training_name,
                    )

        # Save signal_chart once (shared across all runners that save chart data).
        signal_chart_id: int | None = None
        needs_chart_save = any(r.cfg.save_chart_data for r in self._runners)
        if needs_chart_save and ml_candles and self._db:
            # Use the first available ML score as the chart-level label (informational).
            chart_ml_score = next(iter(ml_scores.values()), None) if ml_scores else None
            signal_chart_id = self._db.save_signal_chart(
                symbol=signal.symbol,
                mint=signal.mint_address,
                entry_price=entry_price,
                candles=ml_candles,
                chart_ctx=chart_ctx,
                ml_score=chart_ml_score,
                pair_stats=pair_stats,
                candles_1m=candles if candles else None,
                source_channel=signal.source_channel,
            )

        for runner in self._runners:
            # Resolve this runner's ML score from its designated training strategy + label.
            training_strategy = runner.cfg.ml_training_strategy or runner.cfg.name
            scorer_key = f"{training_strategy}::{runner.cfg.ml_training_label}"
            ml_score = ml_scores.get(scorer_key)

            # ML filter — skip if score is below threshold.
            # If score is None (not enough training data yet), always allow through.
            ml_blocked = (
                runner.cfg.use_ml_filter
                and ml_score is not None
                and ml_score < runner.cfg.ml_min_score
            )
            if ml_blocked:
                logger.info(
                    "[ML_SKIP] [%-12s] %s | score=%.1f < threshold=%.1f | training=%s",
                    runner.name, signal.symbol, ml_score, runner.cfg.ml_min_score,
                    training_strategy,
                )
                signal_log.info(
                    "ML_SKIP    | %-10s | %-44s | score=%.1f | strategy=%s",
                    signal.symbol, signal.mint_address, ml_score, runner.name,
                )

            # Scale up buy size on high-confidence signals (max tier checked first).
            buy_size_override = None
            if runner.cfg.use_ml_filter and ml_score is not None:
                if ml_score >= runner.cfg.ml_max_score_threshold:
                    buy_size_override = runner.cfg.buy_size_usd * runner.cfg.ml_max_size_multiplier
                    logger.info(
                        "[ML_SIZE] [%-12s] %s | score=%.1f >= %.1f — max size $%.2f → $%.2f (%.1fx)",
                        runner.name, signal.symbol, ml_score, runner.cfg.ml_max_score_threshold,
                        runner.cfg.buy_size_usd, buy_size_override, runner.cfg.ml_max_size_multiplier,
                    )
                elif ml_score >= runner.cfg.ml_high_score_threshold:
                    buy_size_override = runner.cfg.buy_size_usd * runner.cfg.ml_size_multiplier
                    logger.info(
                        "[ML_SIZE] [%-12s] %s | score=%.1f >= %.1f — sizing up $%.2f → $%.2f (%.1fx)",
                        runner.name, signal.symbol, ml_score, runner.cfg.ml_high_score_threshold,
                        runner.cfg.buy_size_usd, buy_size_override, runner.cfg.ml_size_multiplier,
                    )

            # ------------------------------------------------------------------
            # Per-signal policy agent (Agent A) — runs when use_policy_agent=True.
            # Evaluates data quality, liquidity, and signal context to apply
            # per-trade overrides: block trade, scale size, or adjust ML score.
            # Any API failure falls back to allowing the trade with no changes.
            # ------------------------------------------------------------------
            policy_blocked = False
            if runner.cfg.use_policy_agent and not ml_blocked:
                signal_context = {
                    "ml_score":              ml_score,
                    "ml_min_score":          runner.cfg.ml_min_score,
                    "used_moralis_10s":      ml_source.startswith("moralis"),
                    "used_birdeye_fallback": ml_source.startswith("birdeye"),
                    "pair_stats_available":  pair_stats is not None,
                    # liquidity_usd / slippage_bps not yet in pair_stats — pass None
                    # so the policy agent skips hard-floor checks for unknown values.
                    "liquidity_usd":         pair_stats.get("liquidity_usd") if pair_stats else None,
                    "slippage_bps":          0,
                }
                try:
                    from trader.agents.policy import propose_policy_decision
                    policy_decision = propose_policy_decision(
                        signal_context=signal_context,
                        strategy=runner.cfg.name,
                    )
                    logger.info(
                        "[POLICY] [%-12s] %s | allow=%s | size_mult=%.2f | score_adj=%.2f | codes=%s",
                        runner.name, signal.symbol,
                        policy_decision["allow_trade"],
                        policy_decision["buy_size_multiplier"],
                        policy_decision["effective_score_adjustment"],
                        policy_decision["reason_codes"],
                    )

                    if not policy_decision["allow_trade"]:
                        policy_blocked = True
                        signal_log.info(
                            "POLICY_BLK | %-10s | %-44s | codes=%s | strategy=%s",
                            signal.symbol, signal.mint_address,
                            policy_decision["reason_codes"], runner.name,
                        )
                    else:
                        # Apply buy size multiplier from policy agent.
                        policy_mult = policy_decision["buy_size_multiplier"]
                        if policy_mult != 1.0:
                            base = buy_size_override if buy_size_override is not None else runner.cfg.buy_size_usd
                            buy_size_override = base * policy_mult
                            logger.info(
                                "[POLICY_SIZE] [%-12s] %s | mult=%.2f → $%.2f",
                                runner.name, signal.symbol, policy_mult, buy_size_override,
                            )
                        # Effective score adjustment — re-check ML floor.
                        score_adj = policy_decision["effective_score_adjustment"]
                        if score_adj != 0.0 and ml_score is not None and runner.cfg.use_ml_filter:
                            adjusted = ml_score + score_adj
                            if adjusted < runner.cfg.ml_min_score:
                                policy_blocked = True
                                logger.info(
                                    "[POLICY_BLK] [%-12s] %s | adj_score=%.1f (%.1f%+.1f) < floor=%.1f",
                                    runner.name, signal.symbol,
                                    adjusted, ml_score, score_adj, runner.cfg.ml_min_score,
                                )

                except Exception:
                    logger.exception(
                        "[POLICY] Agent A call failed for %s — proceeding with defaults", signal.symbol,
                    )

            effective_blocked = ml_blocked or policy_blocked

            # Track skip reason for AI override (set before enter_position).
            _skip_reason: str | None = None
            if ml_blocked:
                _skip_reason = "ML_SKIP"
            elif policy_blocked:
                _skip_reason = "POLICY_BLK"
            elif chart_ctx is not None and not chart_ctx.should_enter:
                _skip_reason = "CHART_SKIP"

            position = None
            if not effective_blocked:
                try:
                    position = runner.enter_position(signal, entry_price, chart_ctx, buy_size_override)
                except Exception:
                    logger.exception(
                        "[ERROR] %s: enter_position failed for %s", runner.name, signal.symbol
                    )
                    continue

            # ------------------------------------------------------------------
            # AI override agent — re-evaluates filtered signals.
            # Active path  (use_ai_override=True):  may force-enter or re-check.
            # Shadow path  (use_ai_override_shadow=True, override=False): calls
            #   agent in background with no trade effect; records SHADOW_* decisions
            #   so we can evaluate agent quality before enabling it live.
            # ------------------------------------------------------------------
            _override_key  = f"{signal.mint_address}:{runner.name}"
            _ai_overrode   = False   # set True when override actually enters a position
            _needs_ai_eval = (
                position is None
                and _skip_reason is not None
                and _override_key not in self._pending_ai_override
                and (runner.cfg.use_ai_override or runner.cfg.use_ai_override_shadow)
            )

            if _needs_ai_eval:
                from trader.agents.ai_override import propose_ai_override, summarize_candles, log_override_decision
                _ai_ctx = {
                    "ml_score":        ml_score,
                    "ml_min_score":    runner.cfg.ml_min_score,
                    "pump_ratio":      chart_ctx.pump_ratio if chart_ctx else None,
                    "pump_ratio_max":  runner.cfg.pump_ratio_max,
                    "vol_trend":       chart_ctx.vol_trend if chart_ctx else None,
                    "chart_reason":    chart_ctx.reason if chart_ctx else None,
                    "ml_source":       ml_source,
                    "source_channel":  signal.source_channel,
                    "pair_stats":      pair_stats,
                    "candles_summary": summarize_candles(ml_candles),
                }

                if runner.cfg.use_ai_override:
                    # Active mode — decision is acted upon immediately
                    try:
                        _ai_decision = propose_ai_override(
                            skip_reason=_skip_reason,
                            signal_context=_ai_ctx,
                            strategy=runner.cfg.name,
                            db_path=self._db.path if self._db else None,
                            training_strategy=runner.cfg.ml_training_strategy or runner.cfg.name,
                        )
                        logger.info(
                            "[AI_OVERRIDE] [%-12s] %s | override=%s | reanalyze=%.0fs | reason=%s",
                            runner.name, signal.symbol,
                            _ai_decision["override"],
                            _ai_decision["reanalyze_after_seconds"],
                            _ai_decision["reason"],
                        )

                        # Determine decision label for DB
                        if _ai_decision["override"]:
                            _decision_label = "OVERRIDE"
                        elif _ai_decision["reanalyze_after_seconds"] > 0:
                            _decision_label = "REANALYZE"
                        else:
                            _decision_label = "REJECT"

                        # Persist decision to DB
                        if self._db:
                            self._db.save_ai_override_decision(
                                strategy=runner.cfg.name,
                                signal_chart_id=signal_chart_id,
                                symbol=signal.symbol,
                                mint=signal.mint_address,
                                skip_reason=_skip_reason,
                                decision=_decision_label,
                                ml_score=ml_score,
                                pump_ratio=chart_ctx.pump_ratio if chart_ctx else None,
                                vol_trend=chart_ctx.vol_trend if chart_ctx else None,
                                agent_reason=_ai_decision["reason"],
                                reanalyze_delay=_ai_decision["reanalyze_after_seconds"],
                            )

                        # Log to file
                        log_override_decision(
                            strategy=runner.cfg.name,
                            symbol=signal.symbol,
                            skip_reason=_skip_reason,
                            decision=_ai_decision,
                            signal_context=_ai_ctx,
                        )

                        if _ai_decision["override"]:
                            position = runner.enter_position(signal, entry_price, None, buy_size_override)
                            if position is not None:
                                _ai_overrode = True
                                signal_log.info(
                                    "AI_OVERRIDE_BUY | %-10s | %-44s | skip_was=%-10s | %s",
                                    signal.symbol, signal.mint_address,
                                    _skip_reason, _ai_decision["reason"],
                                )
                                if self._db:
                                    self._db.log_signal(
                                        "AI_OVERRIDE_BUY", symbol=signal.symbol,
                                        mint=signal.mint_address, strategy=runner.name,
                                        source_channel=signal.source_channel,
                                    )

                        elif _ai_decision["reanalyze_after_seconds"] > 0:
                            _delay = _ai_decision["reanalyze_after_seconds"]
                            self._pending_ai_override.add(_override_key)
                            logger.info(
                                "[AI_OVERRIDE] [%-12s] %s — re-check in %.0fs: %s",
                                runner.name, signal.symbol, _delay, _ai_decision["reason"],
                            )
                            asyncio.create_task(
                                self._ai_override_reanalyze(signal, runner, _delay, _skip_reason)
                            )

                    except Exception:
                        logger.exception(
                            "[AI_OVERRIDE] Agent call failed for %s — skipping override",
                            signal.symbol,
                        )

                else:
                    # Shadow mode — call agent in background, no trade effect
                    asyncio.create_task(
                        self._shadow_override_eval(
                            signal, runner, _skip_reason, _ai_ctx, signal_chart_id,
                        )
                    )

            # Save strategy_outcome for runners that track chart data — even when
            # blocked (ml or policy), these become labeled training examples.
            if runner.cfg.save_chart_data and signal_chart_id is not None:
                outcome_id = self._db.save_strategy_outcome(
                    signal_chart_id=signal_chart_id,
                    strategy=runner.name,
                    entered=position is not None,
                    is_live=runner.cfg.live_trading,
                    source_channel=signal.source_channel,
                    ml_score=ml_score,
                    is_ai_override=_ai_overrode,
                    skip_reason=_skip_reason if position is None else None,
                )
                if position is not None:
                    runner.set_outcome_id(signal.mint_address, outcome_id)
                    runner.set_signal_chart_id(signal.mint_address, signal_chart_id)

        # Schedule reanalysis if chart filter skipped this signal, at least one
        # runner has reanalysis enabled, and this mint isn't already pending.
        # Delay is computed from runner configs (not the shared chart_ctx constant)
        # so each strategy can tune its own reanalysis timing.
        reanalyze_runners = [
            r for r in self._runners
            if r.cfg.use_chart_filter and r.cfg.use_reanalyze
        ]
        if (
            chart_ctx is not None
            and not chart_ctx.should_enter
            and signal.mint_address not in self._pending_reanalysis
            and reanalyze_runners
        ):
            # Determine skip reason from chart metrics
            pumped   = any(chart_ctx.pump_ratio >= r.cfg.pump_ratio_max for r in reanalyze_runners)
            vol_dead = chart_ctx.vol_trend == "DYING"

            if pumped and vol_dead:
                delay = min(r.cfg.reanalyze_both_delay for r in reanalyze_runners)
            elif pumped:
                delay = min(r.cfg.reanalyze_pump_delay for r in reanalyze_runners)
            else:
                delay = min(r.cfg.reanalyze_vol_delay for r in reanalyze_runners)

            self._pending_reanalysis.add(signal.mint_address)
            logger.info(
                "[REANALYZE] %s — scheduled in %.0fs (%.1f min) | reason: %s",
                signal.symbol, delay, delay / 60, chart_ctx.reason,
            )
            asyncio.create_task(self._reanalyze(signal, delay))

    # ------------------------------------------------------------------
    # Reanalysis
    # ------------------------------------------------------------------

    async def _reanalyze(self, signal: TokenSignal, delay: float) -> None:
        """
        Wait `delay` seconds, then re-fetch chart data and decide whether to
        enter the previously skipped signal.

        Only chart-filtered runners are evaluated — baseline runners already
        entered at the original signal time.  One attempt only; if the chart
        still says SKIP the signal is abandoned.
        """
        await asyncio.sleep(delay)
        self._pending_reanalysis.discard(signal.mint_address)

        logger.info("[REANALYZE] %s — re-checking chart after %.0fs", signal.symbol, delay)

        # Fresh price at reanalysis time
        entry_price = await self._birdeye.get_price(signal.mint_address)
        if entry_price is None:
            logger.warning("[REANALYZE] %s — no price available, abandoning", signal.symbol)
            return

        # Fresh OHLCV (current candles, not historical)
        candles = await self._birdeye.get_ohlcv(signal.mint_address, bars=OHLCV_BARS)
        ctx = compute_chart_context(candles, entry_price)

        if ctx is None or not ctx.should_enter:
            reason = ctx.reason if ctx else "no candle data"
            logger.info(
                "[REANALYZE] %s @ $%.8f — still SKIP: %s — abandoning",
                signal.symbol, entry_price, reason,
            )
            signal_log.info(
                "REANALYZE_SKIP | %-10s | %-44s | %s",
                signal.symbol, signal.mint_address, reason,
            )
            return

        logger.info(
            "[REANALYZE] %s @ $%.8f — now BUY: %s — entering chart runners",
            signal.symbol, entry_price, ctx.reason,
        )
        signal_log.info(
            "REANALYZE_BUY  | %-10s | %-44s | %s",
            signal.symbol, signal.mint_address, ctx.reason,
        )

        # Enter only runners with both use_chart_filter and use_reanalyze
        for runner in self._runners:
            if not (runner.cfg.use_chart_filter and runner.cfg.use_reanalyze):
                continue
            try:
                runner.enter_position(signal, entry_price, ctx)
            except Exception:
                logger.exception(
                    "[ERROR] %s: reanalysis enter_position failed for %s",
                    runner.name, signal.symbol,
                )

    # ------------------------------------------------------------------
    # AI override reanalysis
    # ------------------------------------------------------------------

    async def _ai_override_reanalyze(
        self,
        signal,
        runner: StrategyRunner,
        delay: float,
        original_skip: str,
    ) -> None:
        """
        Wait `delay` seconds, re-fetch all signal data, and call the AI override
        agent again.  If it overrides, force-enter the position.  One attempt only.
        """
        await asyncio.sleep(delay)
        override_key = f"{signal.mint_address}:{runner.name}"
        self._pending_ai_override.discard(override_key)

        logger.info(
            "[AI_OVERRIDE] [%-12s] %s — re-checking after %.0fs (original skip: %s)",
            runner.name, signal.symbol, delay, original_skip,
        )

        entry_price = await self._birdeye.get_price(signal.mint_address)
        if entry_price is None:
            logger.warning("[AI_OVERRIDE] %s — no price at re-check, abandoning", signal.symbol)
            return

        candles = await self._birdeye.get_ohlcv(signal.mint_address, bars=OHLCV_BARS)
        chart_ctx = compute_chart_context(candles, entry_price)

        ml_candles: list = []
        ml_source = "none"
        if self._moralis:
            try:
                ml_candles = await self._moralis.get_ohlcv(
                    signal.mint_address, bars=MORALIS_OHLCV_BARS, interval=MORALIS_OHLCV_INTERVAL,
                )
                if ml_candles:
                    ml_source = f"moralis/{MORALIS_OHLCV_INTERVAL}"
            except Exception:
                pass
        if not ml_candles:
            try:
                ml_candles = await self._birdeye.get_ohlcv(
                    signal.mint_address, bars=ML_OHLCV_BARS, interval=ML_OHLCV_INTERVAL,
                )
                if ml_candles:
                    ml_source = f"birdeye/{ML_OHLCV_INTERVAL}"
            except Exception:
                pass

        pair_stats = None
        if self._moralis:
            try:
                pair_stats = await self._moralis.get_pair_stats(signal.mint_address)
            except Exception:
                pass

        training_strategy = runner.cfg.ml_training_strategy or runner.cfg.name
        scorer_key = f"{training_strategy}::{runner.cfg.ml_training_label}"
        ml_score = None
        if scorer_key in self._ml_scorers and ml_candles:
            use_moralis = scorer_key in self._ml_prefer_moralis
            score_candles = ml_candles if use_moralis else (candles or ml_candles)
            ml_score = self._ml_scorers[scorer_key].score(
                score_candles, chart_ctx=chart_ctx, pair_stats=pair_stats,
                source_channel=signal.source_channel,
            )

        try:
            from trader.agents.ai_override import propose_ai_override, summarize_candles
            ai_ctx = {
                "ml_score":        ml_score,
                "ml_min_score":    runner.cfg.ml_min_score,
                "pump_ratio":      chart_ctx.pump_ratio if chart_ctx else None,
                "pump_ratio_max":  runner.cfg.pump_ratio_max,
                "vol_trend":       chart_ctx.vol_trend if chart_ctx else None,
                "chart_reason":    chart_ctx.reason if chart_ctx else None,
                "ml_source":       ml_source,
                "source_channel":  signal.source_channel,
                "pair_stats":      pair_stats,
                "candles_summary": summarize_candles(ml_candles),
            }
            ai_decision = propose_ai_override(
                skip_reason=f"RECHECK_{original_skip}",
                signal_context=ai_ctx,
                strategy=runner.cfg.name,
                db_path=self._db.path if self._db else None,
                training_strategy=runner.cfg.ml_training_strategy or runner.cfg.name,
            )
            logger.info(
                "[AI_OVERRIDE_RECHECK] [%-12s] %s | override=%s | reason=%s",
                runner.name, signal.symbol, ai_decision["override"], ai_decision["reason"],
            )

            if not ai_decision["override"]:
                signal_log.info(
                    "AI_OVERRIDE_SKIP | %-10s | %-44s | recheck of %-10s | %s",
                    signal.symbol, signal.mint_address, original_skip, ai_decision["reason"],
                )
                return

            position = runner.enter_position(signal, entry_price, None, None)
            if position is not None:
                signal_log.info(
                    "AI_OVERRIDE_BUY | %-10s | %-44s | recheck of %-10s | %s",
                    signal.symbol, signal.mint_address, original_skip, ai_decision["reason"],
                )
                if self._db:
                    self._db.log_signal(
                        "AI_OVERRIDE_BUY", symbol=signal.symbol,
                        mint=signal.mint_address, strategy=runner.name,
                        source_channel=signal.source_channel,
                    )

        except Exception:
            logger.exception(
                "[AI_OVERRIDE] Re-check agent call failed for %s", signal.symbol,
            )

    async def _shadow_override_eval(
        self,
        signal,
        runner: StrategyRunner,
        skip_reason: str,
        ai_ctx: dict,
        signal_chart_id: int | None,
    ) -> None:
        """
        Shadow mode: call the AI override agent with no trade effect.
        Records SHADOW_OVERRIDE / SHADOW_REJECT / SHADOW_REANALYZE in
        ai_override_decisions so we can compare to what actually happened.
        """
        try:
            from trader.agents.ai_override import propose_ai_override, log_override_decision
            decision = propose_ai_override(
                skip_reason=skip_reason,
                signal_context=ai_ctx,
                strategy=runner.cfg.name,
                db_path=self._db.path if self._db else None,
                training_strategy=runner.cfg.ml_training_strategy or runner.cfg.name,
            )

            if decision["override"]:
                label = "SHADOW_OVERRIDE"
            elif decision["reanalyze_after_seconds"] > 0:
                label = "SHADOW_REANALYZE"
            else:
                label = "SHADOW_REJECT"

            logger.info(
                "[AI_SHADOW] [%-12s] %s | would=%s | reason=%s",
                runner.name, signal.symbol, label, decision["reason"],
            )

            if self._db:
                self._db.save_ai_override_decision(
                    strategy=runner.cfg.name,
                    signal_chart_id=signal_chart_id,
                    symbol=signal.symbol,
                    mint=signal.mint_address,
                    skip_reason=skip_reason,
                    decision=label,
                    ml_score=ai_ctx.get("ml_score"),
                    pump_ratio=ai_ctx.get("pump_ratio"),
                    vol_trend=ai_ctx.get("vol_trend"),
                    agent_reason=decision["reason"],
                    reanalyze_delay=decision["reanalyze_after_seconds"],
                )

            log_override_decision(
                strategy=runner.cfg.name,
                symbol=signal.symbol,
                skip_reason=skip_reason,
                decision=decision,
                signal_context=ai_ctx,
                shadow=True,
            )

        except Exception:
            logger.exception(
                "[AI_SHADOW] Agent call failed for %s", signal.symbol,
            )

    # ------------------------------------------------------------------
    # Monitoring loop
    # ------------------------------------------------------------------

    async def monitor_positions(self, cycles: Optional[int] = None) -> None:
        """
        Shared price polling loop.

        Each tick:
            1. Collect unique mints across all runners' open positions.
            2. Fetch each mint's price exactly once from Birdeye.
            3. Distribute each price to every runner holding that mint.
            4. Each runner evaluates its own exit logic independently.

        A mint is tracked until ALL strategies have exited it.
        """
        if self._cfg.dry_run:
            logger.info("[DRY_RUN] Monitor loop disabled — outcomes simulated by price_history.py")
            return

        mode = f"{cycles} cycles" if cycles is not None else "live (∞)"
        logger.info(
            "[MONITOR] Starting | strategies=%d | mode=%s | interval=%.1fs",
            len(self._runners), mode, self._cfg.poll_interval_seconds,
        )

        tick = 0
        while cycles is None or tick < cycles:
            tick += 1

            # Map: mint → [runners that currently hold it]
            mint_to_runners: dict[str, list[StrategyRunner]] = {}
            for runner in self._runners:
                for pos in runner.get_open_positions():
                    mint_to_runners.setdefault(pos.mint_address, []).append(runner)

            if not mint_to_runners:
                logger.debug("[MONITOR] Tick %d — no open positions", tick)
                await asyncio.sleep(self._cfg.poll_interval_seconds)
                continue

            # One batch call covers every unique mint across all strategies
            prices = await self._birdeye.get_prices_batch(list(mint_to_runners.keys()))

            # Fan each price out to all runners holding that mint
            for mint, holders in mint_to_runners.items():
                price = prices.get(mint)
                if price is None:
                    logger.warning("[PRICE] No price for %s this tick", mint)
                    continue

                for runner in holders:
                    pos = runner.get_position(mint)
                    if pos is None or pos.status == "CLOSED":
                        continue

                    pos.last_price = price
                    logger.info(
                        "[PRICE] [%-12s] %-10s | $%.8f | tp=$%.8f | sl=$%.8f | trailing=%s%s",
                        runner.name, pos.symbol, price,
                        pos.take_profit_price, pos.stop_loss_price,
                        pos.trailing_active,
                        f" | trail_stop=${pos.trailing_stop_price:.8f}" if pos.trailing_stop_price else "",
                    )

                    try:
                        runner.evaluate_position(pos, price)
                    except Exception:
                        logger.exception(
                            "[ERROR] %s: evaluate_position failed for %s",
                            runner.name, pos.symbol,
                        )

            await asyncio.sleep(self._cfg.poll_interval_seconds)

        logger.info("[MONITOR] Complete after %d ticks", tick)
        self.print_summary()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_summary(self, current_prices: dict[str, float] | None = None) -> None:
        """Emit per-strategy summaries followed by a global aggregate."""
        sep = "=" * 72
        summaries = [r.summary(current_prices) for r in self._runners]

        for s in summaries:
            logger.info(sep)
            logger.info("[SUMMARY]  Strategy: %s", s["name"])
            logger.info(sep)
            logger.info("  Starting capital   : $%.2f", s["starting_cash"])
            logger.info("  Available cash     : $%.2f", s["available_cash"])
            logger.info("  Open market value  : $%.4f", s["market_value"])
            logger.info("  Total equity       : $%.4f", s["equity"])
            logger.info("  " + "-" * 52)
            logger.info("  Positions — open   : %d", s["open_count"])
            logger.info("  Positions — closed : %d", s["closed_count"])
            logger.info("  Win rate           : %.1f%%", s["win_rate"])
            logger.info("  " + "-" * 52)
            logger.info("  Realized PnL       : $%+.4f", s["realized_pnl"])
            logger.info("  Unrealized PnL     : $%+.4f", s["unrealized_pnl"])
            logger.info("  Net PnL vs start   : $%+.4f  (%+.2f%%)", s["net_pnl"], s["net_pnl_pct"])
            logger.info(sep)

            for p in s["positions"]:
                unreal = 0.0
                if p.status == "OPEN" and p.last_price:
                    unreal = (p.last_price - p.entry_price) * p.remaining_quantity
                logger.info(
                    "  [%-6s] %-10s | entry=$%.6f | last=$%.6f | "
                    "realized=$%+.4f | unrealized=$%+.4f | trail=%s | exit=%s",
                    p.status, p.symbol,
                    p.entry_price, p.last_price or 0.0,
                    p.realized_pnl_usd, unreal,
                    p.trailing_active,
                    p.sell_reason or "—",
                )
            logger.info(sep)

        # Global aggregate (only printed when more than one strategy is running)
        if len(summaries) > 1:
            total_start   = sum(s["starting_cash"]  for s in summaries)
            total_cash    = sum(s["available_cash"]  for s in summaries)
            total_mv      = sum(s["market_value"]    for s in summaries)
            total_equity  = sum(s["equity"]          for s in summaries)
            total_realized   = sum(s["realized_pnl"]   for s in summaries)
            total_unrealized = sum(s["unrealized_pnl"]  for s in summaries)
            total_net     = total_equity - total_start
            total_net_pct = (total_net / total_start * 100) if total_start else 0.0
            total_open    = sum(s["open_count"]   for s in summaries)
            total_closed  = sum(s["closed_count"] for s in summaries)

            logger.info(sep)
            logger.info("[SUMMARY]  GLOBAL  (%d strategies)", len(self._runners))
            logger.info(sep)
            logger.info("  Total starting capital : $%.2f", total_start)
            logger.info("  Total cash             : $%.2f", total_cash)
            logger.info("  Total open value       : $%.4f", total_mv)
            logger.info("  Total equity           : $%.4f", total_equity)
            logger.info("  " + "-" * 52)
            logger.info("  Total open positions   : %d", total_open)
            logger.info("  Total closed positions : %d", total_closed)
            logger.info("  " + "-" * 52)
            logger.info("  Total realized PnL     : $%+.4f", total_realized)
            logger.info("  Total unrealized PnL   : $%+.4f", total_unrealized)
            logger.info("  Total net PnL          : $%+.4f  (%+.2f%%)", total_net, total_net_pct)
            logger.info(sep)
