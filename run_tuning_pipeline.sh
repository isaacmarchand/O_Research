#!/bin/bash

# ==============================================================================
# FPCA Sequential Tuning and Estimation Pipeline for First 3 FPCs
# Prevent Mac sleep during execution using caffeinate
# ==============================================================================

# Re-exec under caffeinate to prevent sleep if not already running under it
if [ "$1" != "--caffeinated" ]; then
    echo "=========================================================="
    echo "Starting tuning pipeline under caffeinate (preventing sleep)..."
    echo "=========================================================="
    exec caffeinate -s -i "$0" --caffeinated "$@"
fi

# Shift the internal '--caffeinated' flag
shift

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# PIPELINE CONFIGURATION (Modify arguments below as needed)
# ------------------------------------------------------------------------------
PYTHON_BIN="/opt/anaconda3/envs/O_Research/bin/python"
DATA_PATH="data/DJX_data/DJX_lists_traded.pkl"
N_DAYS=""                       # Number of days to limit (empty to use all days)

# Cross-Validation parameters
N_SPLITS=5
TRAIN_WINDOW_SIZE=2000           # Training window size (empty ("") for expanding window)
TEST_WINDOW_SIZE=30            # Validation window size (empty ("") for default)
THETA_CAL=1.0                   # Calendar arbitrage penalty weight
THETA_BUT=1.0                   # Butterfly arbitrage penalty weight
FRICTION_TOL=0.001               # Tolerance for arbitrage check

# Grid search spaces (space-separated lists)
OMEGA_M_GRID="0.025 0.05 0.1 0.2 .5"
OMEGA_M2_GRID="0.0 0.01 0.025 .05 .1"
OMEGA_T_GRID="0.005 0.01 0.025 0.05"

# Spline basis specifications
NB_SPLINE_MONEYNESS=30
NB_SPLINE_TAU=36
ORDER_MONEYNESS=4
ORDER_TAU=4



# Fit/Optimization iterations
MAXIT=30
THRESHOLD=1e-4
OUTPUT_PREFIX="cv_tuning_sequential"

# ------------------------------------------------------------------------------
# BASH INITIALIZATION AND ARGUMENT ASSEMBLY
# ------------------------------------------------------------------------------
set -e # Exit immediately if a command exits with a non-zero status

# Print configuration summary
echo "Pipeline Configuration:"
echo "  Python Bin: $PYTHON_BIN"
echo "  Dataset: $DATA_PATH"
echo "  n_days: ${N_DAYS:-All}"
echo "  cv_type: rolling, n_splits: $N_SPLITS"
echo "  Arbitrage Penalties: theta_cal=$THETA_CAL, theta_but=$THETA_BUT, friction_tol=$FRICTION_TOL"
echo "  Moneyness grid: $OMEGA_M_GRID"
echo "  Moneyness2 grid: $OMEGA_M2_GRID"
echo "  Tau/maturity grid: $OMEGA_T_GRID"
echo "--------------------------------------------------------"

# Assemble tuning arguments array
TUNE_ARGS=(
    "--data_path" "$DATA_PATH"
    "--n_splits" "$N_SPLITS"
    "--maxit" "$MAXIT"
    "--threshold" "$THRESHOLD"
    "--theta_cal" "$THETA_CAL"
    "--theta_but" "$THETA_BUT"
    "--friction_tol" "$FRICTION_TOL"
    "--output_prefix" "$OUTPUT_PREFIX"
)

if [ -n "$N_DAYS" ]; then
    TUNE_ARGS+=( "--n_days" "$N_DAYS" )
fi
if [ -n "$TRAIN_WINDOW_SIZE" ]; then
    TUNE_ARGS+=( "--train_window_size" "$TRAIN_WINDOW_SIZE" )
fi
if [ -n "$TEST_WINDOW_SIZE" ]; then
    TUNE_ARGS+=( "--test_window_size" "$TEST_WINDOW_SIZE" )
fi

# Grid values
TUNE_ARGS+=( "--omega_m_grid" )
for val in $OMEGA_M_GRID; do TUNE_ARGS+=( "$val" ); done

TUNE_ARGS+=( "--omega_m2_grid" )
for val in $OMEGA_M2_GRID; do TUNE_ARGS+=( "$val" ); done

TUNE_ARGS+=( "--omega_t_grid" )
for val in $OMEGA_T_GRID; do TUNE_ARGS+=( "$val" ); done

# Assemble basis model arguments array
MODEL_ARGS=(
    "--nb_spline_moneyness" "$NB_SPLINE_MONEYNESS"
    "--nb_spline_tau" "$NB_SPLINE_TAU"
    "--order_moneyness" "$ORDER_MONEYNESS"
    "--order_tau" "$ORDER_TAU"
)

# Output pickle paths
FPC0_OUT="data/DJX_data/fpc0_fitted.pkl"
FPC1_OUT="data/DJX_data/fpc1_fitted.pkl"
FPC2_OUT="data/DJX_data/fpc2_fitted.pkl"

# Helper function to parse JSON results for best hyperparameters
get_best_param() {
    local json_file="$1"
    local key="$2"
    "$PYTHON_BIN" -c "import json; print(json.load(open('$json_file'))['$key'])"
}

# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================

# --- FPC 1 (fpc_index = 0) ---
echo -e "\n>>> STEP 1: Tuning FPC 1..."
"$PYTHON_BIN" tune_smoothness.py "${TUNE_ARGS[@]}" "${MODEL_ARGS[@]}" --fpc_index 0

JSON_FPC1="${OUTPUT_PREFIX}_fpc1_best_params.json"
omega_m=$(get_best_param "$JSON_FPC1" "best_omega_m")
omega_m2=$(get_best_param "$JSON_FPC1" "best_omega_m2")
omega_t=$(get_best_param "$JSON_FPC1" "best_omega_t")

echo ">>> STEP 2: Estimating & Saving FPC 1 with optimal (omega_m=$omega_m, omega_m2=$omega_m2, omega_t=$omega_t)..."
"$PYTHON_BIN" estimate_fpc.py \
    --data_path "$DATA_PATH" \
    ${N_DAYS:+--n_days "$N_DAYS"} \
    --fpc_index 0 \
    --output_path "$FPC0_OUT" \
    --maxit "$MAXIT" \
    --threshold "$THRESHOLD" \
    --omega_m "$omega_m" \
    --omega_m2 "$omega_m2" \
    --omega_t "$omega_t" \
    "${MODEL_ARGS[@]}"


# --- FPC 2 (fpc_index = 1) ---
echo -e "\n>>> STEP 3: Tuning FPC 2..."
"$PYTHON_BIN" tune_smoothness.py "${TUNE_ARGS[@]}" "${MODEL_ARGS[@]}" --fpc_index 1 --previous_blist_path "$FPC0_OUT"

JSON_FPC2="${OUTPUT_PREFIX}_fpc2_best_params.json"
omega_m=$(get_best_param "$JSON_FPC2" "best_omega_m")
omega_m2=$(get_best_param "$JSON_FPC2" "best_omega_m2")
omega_t=$(get_best_param "$JSON_FPC2" "best_omega_t")

echo ">>> STEP 4: Estimating & Saving FPC 2 with optimal (omega_m=$omega_m, omega_m2=$omega_m2, omega_t=$omega_t)..."
"$PYTHON_BIN" estimate_fpc.py \
    --data_path "$DATA_PATH" \
    ${N_DAYS:+--n_days "$N_DAYS"} \
    --fpc_index 1 \
    --previous_path "$FPC0_OUT" \
    --output_path "$FPC1_OUT" \
    --maxit "$MAXIT" \
    --threshold "$THRESHOLD" \
    --omega_m "$omega_m" \
    --omega_m2 "$omega_m2" \
    --omega_t "$omega_t" \
    "${MODEL_ARGS[@]}"


# --- FPC 3 (fpc_index = 2) ---
echo -e "\n>>> STEP 5: Tuning FPC 3..."
"$PYTHON_BIN" tune_smoothness.py "${TUNE_ARGS[@]}" "${MODEL_ARGS[@]}" --fpc_index 2 --previous_blist_path "$FPC1_OUT"

JSON_FPC3="${OUTPUT_PREFIX}_fpc3_best_params.json"
omega_m=$(get_best_param "$JSON_FPC3" "best_omega_m")
omega_m2=$(get_best_param "$JSON_FPC3" "best_omega_m2")
omega_t=$(get_best_param "$JSON_FPC3" "best_omega_t")

echo ">>> STEP 6: Estimating & Saving FPC 3 with optimal (omega_m=$omega_m, omega_m2=$omega_m2, omega_t=$omega_t)..."
"$PYTHON_BIN" estimate_fpc.py \
    --data_path "$DATA_PATH" \
    ${N_DAYS:+--n_days "$N_DAYS"} \
    --fpc_index 2 \
    --previous_path "$FPC1_OUT" \
    --output_path "$FPC2_OUT" \
    --maxit "$MAXIT" \
    --threshold "$THRESHOLD" \
    --omega_m "$omega_m" \
    --omega_m2 "$omega_m2" \
    --omega_t "$omega_t" \
    "${MODEL_ARGS[@]}"

echo -e "\n=========================================================="
echo "Pipeline executed successfully! Output files:"
echo "  FPC 1: $FPC0_OUT"
echo "  FPC 2: $FPC1_OUT"
echo "  FPC 3: $FPC2_OUT"
echo "=========================================================="
