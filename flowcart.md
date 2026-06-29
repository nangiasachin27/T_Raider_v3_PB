graph TD
    %% Styling
    classDef brain fill:#0F172A,stroke:#38BDF8,color:#F8FAFC,stroke-width:2px;
    classDef file fill:#F1F5F9,stroke:#94A3B8,color:#334155,stroke-width:1px;
    classDef muscle fill:#0284C7,stroke:#0EA5E9,color:#F8FAFC,stroke-width:2px;
    classDef gate fill:#DC2626,stroke:#F87171,color:#F8FAFC,stroke-width:2px;
    classDef active fill:#16A34A,stroke:#4ADE80,color:#F8FAFC,stroke-width:2px;

    %% STAGE 1: CALIBRATION
    subgraph STAGE_1 [Stage 1: Calibration & Optimization]
        A[auto_optimizer.py]:::brain -->|1. Slices Folds| B(Walk-Forward Matrix):::brain
        B -->|2. Resamples Trades| C(Monte Carlo 5000x Filter):::brain
        C -->|3. Compiles Param Grids| D(Composite Scoring Module):::brain
    end
    D -->|4. Serializes Brains| E[(optimal_params.json)]:::file

    %% STAGE 2: DAILY ORCHESTRATION
    subgraph STAGE_2 [Stage 2: Orchestration & Timeline Management]
        F[quarterly_manager.py]:::muscle -->|5. Asserts Hours| G{Trading Guard Open?}:::gate
        G -->|No| G_Exit(Graceful Exit 0)
        G -->|Yes| H(HarvestEngine Trigger Check):::muscle
        H -->|True| I(Harvest Engine: Liquidate All):::muscle
        I -->|Update Setup| J[(quarterly_config.json)]:::file
        H -->|False| K(Active Profit Engine Tracking):::active
    end
    K -->|6. Scale Sizing Risk| L[(config/.active_risk_mult.json)]:::file

    %% STAGE 3: INTERDAY TRADING LOOP
    subgraph STAGE_3 [Stage 3: Risk Evaluation & Sizing Engine]
        M[autopilot/bot.py]:::muscle -->|7. Evacuate Risk| N(Phase 0: Real-Time Risk Exits):::gate
        N -->|Adaptive Trailing Stops| N1[check_trailing_stops]:::gate
        N -->|Per-Stock Targets| N2[check_profit_targets]:::gate
        N -->|45-Day Stagnation| N3[check_dead_money_exits]:::gate
        N -->|10% Account Floor| N4[check_stop_losses]:::gate
    end
    L --> M
    E --> M

    %% STAGE 4: SCREENING GATES
    subgraph STAGE_4 [Stage 4: Screening & Multi-Layer Filters]
        P[daily_screener.py]:::brain -->|8. Market Trend| Q{Gate 1: Regime Close > EMA50?}:::gate
        Q -->|Bear Market Pullbacks| Q1(Nifty Drop 50d High Check):::gate
        Q -->|Pass| R{Gate 1b: MacroFilter checks VIX & Flow}:::gate
        R -->|Pass| S(Gate 2: Volume Confirmation >= 80%):::gate
        S -->|Pass| T{Gate 4.5 & 4.6: Correlation & Beta Gates}:::gate
        T -->|Pass| U(Factor Z-Score & Sector Ranking):::brain
    end
    N4 -->|Query Opportunities| P
    U -->|9. Actionable Signals List| V(Phase 2: Sized Entry Queue):::muscle
    V -->|10. Execute & Log Trade| W[(config/portfolio.json)]:::file

    %% Set classes
    class A,B,C,D,P,U brain;
    class E,J,L,W file;
    class F,H,I,M,V muscle;
    class G,N,N1,N2,N3,N4,Q,Q1,R,S,T gate;
    class K active;