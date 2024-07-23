#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..    # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                        # Do not modify

# ************************** Customizable Arguments ************************************

ARGUMENT=$DATA_ROOT/arguments.txt

# --------------------------------------------------------------------------------------

# OPTIONAL_ARGUMENT=false

# **************************************************************************************

cmd=( python3 "$PROJECT_ROOT/main.py" \
  --argument "$ARGUMENT" )

if [[ -v OPTIONAL_ARGUMENT ]]; then
  cmd+=( --optional_argument "$OPTIONAL_ARGUMENT" )
fi

"${cmd[@]}"
