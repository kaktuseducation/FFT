# arb_patch_v10 — how to apply and run

Check you have the right archive FIRST:

    cat OddFFT/PATCH_VERSION      # must say v10

## Unpack (from the directory that CONTAINS OddFFT/)

    tar xzf arb_patch_v10.tar.gz

It overwrites files inside OddFFT/ and adds nothing you have to wire up.

## Run

    cd OddFFT
    bash tests/run_tests.sh       # host-only, no CANN needed, must end ALL TESTS PASS
    bash run_matrix.sh -v "$(./build/oddfft --list | tail -n +2 | awk -F, '$1=="mix"{print $1"/"$2}' | tr '\n' ' ')"

or simply, for the whole table:

    bash run_matrix.sh

## What to read in the log, in this order

1. `[CALIB]` — it must now print two times and a decision, e.g.

        [CALIB] mix/frank_hi_ps_xs N=100800 tile=64 minT=8: unbound=1693.48 us  bound=1601.20 us
        [CALIB] -> OMP_PROC_BIND=close

   If it says "no usable [RESULT] line" it now also prints six lines of the
   calibration run's own log underneath. Send those: they say why.

2. `[INFO] OpenMP as used for this table:` — the setting every cell ran under.

3. `[DONE ] cells=N  PASS=...` and the analyzer's `cells:` line — these two
   numbers MUST now agree. In v9 they read 144 and 146.

4. Sanity, before believing any of it: a paired speedup above roughly 20x, or
   a cell where one AIV core beats forty, means the host was busy during the
   run. Then the table is noise and has to be taken again, not interpreted.
