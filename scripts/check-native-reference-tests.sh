#!/usr/bin/env bash
# Exact-name gate for the native reference test targets.
#
# Every test listed in the manifest must exist under its exact name, run, and
# pass; the target must contain no unlisted test; and libtest must report zero
# failed, ignored and filtered-out tests. A target that executes zero tests
# fails. See docs/native-reference-tests.md.
#
# Usage:
#   scripts/check-native-reference-tests.sh [--manifest PATH]
#   scripts/check-native-reference-tests.sh --self-test
#
# Dependencies: bash and cargo only (no Python, no jq). Portable to bash 3.2
# (macOS /bin/bash) and to BSD and GNU userlands: no associative arrays, no
# mapfile/readarray, no ${x,,}, no grep -P, no suffixless sed -i, no sort -V.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_MANIFEST="$REPO_ROOT/scripts/native-reference-manifest.txt"

# Protected targets: each must keep at least one manifest entry, so a target
# cannot silently leave the gate by having all of its lines deleted.
REQUIRED_TARGETS="thresh-association/mht_reference thresh-filter/imm_reference thresh-filter/imm_current_policy"

WORK_DIR=""
cleanup() {
    if [ -n "$WORK_DIR" ]; then
        rm -rf "$WORK_DIR"
    fi
}
trap cleanup EXIT

make_work_dir() {
    WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/native-reference.XXXXXX")"
}

complain() {
    echo "native reference gate: $*" >&2
}

indent() {
    sed 's/^/    /' >&2
}

# validate_manifest MANIFEST OUT
# Writes normalised "<package> <target> <test>" entries to OUT. Fails on a
# missing, empty, malformed or duplicated manifest, or on a missing required
# target. '#' starts a comment; blank lines are ignored.
validate_manifest() {
    local manifest="$1" out="$2" bad dups target
    if [ ! -f "$manifest" ]; then
        complain "manifest not found: $manifest"
        return 1
    fi
    bad="$(awk '{
        sub(/#.*/, "")
        if (NF == 0) next
        if (NF != 3 || $1 !~ /^[A-Za-z0-9_-]+$/ || $2 !~ /^[A-Za-z0-9_-]+$/ || $3 !~ /^[A-Za-z0-9_:]+$/)
            print "line " NR ": " $0
    }' "$manifest")"
    if [ -n "$bad" ]; then
        complain "malformed manifest entries (expected '<package> <target> <test>'):"
        printf '%s\n' "$bad" | indent
        return 1
    fi
    awk '{ sub(/#.*/, ""); if (NF) print $1, $2, $3 }' "$manifest" >"$out"
    if [ ! -s "$out" ]; then
        complain "manifest has no entries: $manifest"
        return 1
    fi
    dups="$(LC_ALL=C sort "$out" | uniq -d)"
    if [ -n "$dups" ]; then
        complain "duplicated manifest entries:"
        printf '%s\n' "$dups" | indent
        return 1
    fi
    for target in $REQUIRED_TARGETS; do
        if ! awk -v want="$target" '($1 "/" $2) == want { found = 1 } END { exit found ? 0 : 1 }' "$out"; then
            complain "manifest has no entries for required target $target"
            return 1
        fi
    done
}

# check_inventory LABEL EXPECTED LISTED
# EXPECTED and LISTED are sorted name files. Fails on missing or unlisted tests.
check_inventory() {
    local label="$1" expected="$2" listed="$3" missing unlisted status=0
    missing="$(LC_ALL=C comm -23 "$expected" "$listed")"
    unlisted="$(LC_ALL=C comm -13 "$expected" "$listed")"
    if [ -n "$missing" ]; then
        complain "$label: manifest tests missing from the target (removed or renamed?):"
        printf '%s\n' "$missing" | indent
        status=1
    fi
    if [ -n "$unlisted" ]; then
        complain "$label: target tests not in the manifest:"
        printf '%s\n' "$unlisted" | indent
        status=1
    fi
    return "$status"
}

# check_results LABEL EXPECTED RUN_OUTPUT COUNT
# Every expected test must report ok (a should-panic test reports
# "test NAME - should panic ... ok"); no test may report FAILED or ignored;
# and there must be exactly one summary reporting COUNT passed with zero
# failed, ignored, measured and filtered-out tests.
check_results() {
    local label="$1" expected="$2" run_out="$3" count="$4" passed not_passed bad summaries n_summaries want status=0
    passed="$(sed -n -E 's/^test ([^ ]+)( - should panic)? \.\.\. ok$/\1/p' "$run_out" | LC_ALL=C sort)"
    not_passed="$(printf '%s\n' "$passed" | LC_ALL=C comm -23 "$expected" -)"
    if [ -n "$not_passed" ]; then
        complain "$label: manifest tests that did not report ok:"
        printf '%s\n' "$not_passed" | indent
        status=1
    fi
    bad="$(grep -E '^test [^ ]+( - should panic)? \.\.\. (FAILED|ignored)' "$run_out" || true)"
    if [ -n "$bad" ]; then
        complain "$label: failed or ignored tests:"
        printf '%s\n' "$bad" | indent
        status=1
    fi
    summaries="$(grep -E '^test result: ' "$run_out" || true)"
    n_summaries="$(printf '%s\n' "$summaries" | grep -c '^test result: ' || true)"
    want="test result: ok. $count passed; 0 failed; 0 ignored; 0 measured; 0 filtered out"
    if [ "$n_summaries" -ne 1 ]; then
        complain "$label: expected exactly one libtest summary, found $n_summaries"
        status=1
    else
        case "$summaries" in
            "$want" | "$want;"*) ;;
            *)
                complain "$label: unexpected summary (want '$want'):"
                printf '%s\n' "$summaries" | indent
                status=1
                ;;
        esac
    fi
    return "$status"
}

# check_target LABEL EXPECTED_UNSORTED LIST_OUTPUT RUN_OUTPUT SCRATCH
check_target() {
    local label="$1" expected_raw="$2" list_out="$3" run_out="$4" scratch="$5" count status=0
    LC_ALL=C sort "$expected_raw" >"$scratch/expected.sorted"
    sed -n 's/: test$//p' "$list_out" | LC_ALL=C sort >"$scratch/listed.sorted"
    count="$(wc -l <"$scratch/expected.sorted" | tr -d ' ')"
    if [ "$count" -eq 0 ]; then
        complain "$label: no expected tests"
        return 1
    fi
    check_inventory "$label" "$scratch/expected.sorted" "$scratch/listed.sorted" || status=1
    check_results "$label" "$scratch/expected.sorted" "$run_out" "$count" || status=1
    return "$status"
}

# run_cargo_target PACKAGE TARGET DIR
# Writes DIR/list, DIR/run and their stderr. Fails only if listing fails.
run_cargo_target() {
    local package="$1" target="$2" dir="$3"
    if ! (cd "$REPO_ROOT" && cargo test --locked -p "$package" --no-default-features --test "$target" \
        -- --list --format terse) <"/dev/null" >"$dir/list" 2>"$dir/list.err"; then
        complain "$package/$target: cargo could not list the target"
        tail -n 20 "$dir/list.err" | indent
        return 1
    fi
    (cd "$REPO_ROOT" && cargo test --locked -p "$package" --no-default-features --test "$target" \
        -- --format pretty --color never) <"/dev/null" >"$dir/run" 2>"$dir/run.err" || true
}

run_gate() {
    local manifest="$1" package target dir count total=0 n_targets=0 status=0
    make_work_dir
    validate_manifest "$manifest" "$WORK_DIR/entries" || return 1
    awk '!seen[$1 " " $2]++ { print $1, $2 }' "$WORK_DIR/entries" >"$WORK_DIR/targets"
    while read -r package target; do
        dir="$WORK_DIR/$package.$target"
        mkdir -p "$dir"
        awk -v p="$package" -v t="$target" '$1 == p && $2 == t { print $3 }' "$WORK_DIR/entries" >"$dir/expected"
        count="$(wc -l <"$dir/expected" | tr -d ' ')"
        n_targets=$((n_targets + 1))
        if ! run_cargo_target "$package" "$target" "$dir"; then
            status=1
            continue
        fi
        if check_target "$package/$target" "$dir/expected" "$dir/list" "$dir/run" "$dir"; then
            echo "ok: $package/$target: $count required tests passed"
            total=$((total + count))
        else
            tail -n 20 "$dir/run.err" | indent
            status=1
        fi
    done <"$WORK_DIR/targets"
    if [ "$status" -ne 0 ]; then
        complain "FAILED"
        return 1
    fi
    echo "Native reference gate PASSED: $total required tests across $n_targets targets; zero ignored."
}

# ---------------------------------------------------------------- self-test

SELF_TEST_CASES=0
SELF_TEST_FAILURES=0

expect() {
    local want="$1" name="$2" got
    shift 2
    if "$@" 2>"$WORK_DIR/self-test.err"; then got=pass; else got=fail; fi
    SELF_TEST_CASES=$((SELF_TEST_CASES + 1))
    if [ "$got" = "$want" ] && [ "$got" = fail ]; then
        echo "self-test ok: $name (fail: $(head -n 1 "$WORK_DIR/self-test.err" | sed 's/^native reference gate: //'))"
    elif [ "$got" = "$want" ]; then
        echo "self-test ok: $name (pass)"
    else
        echo "self-test MISMATCH: $name expected $want, got $got" >&2
        indent <"$WORK_DIR/self-test.err"
        SELF_TEST_FAILURES=$((SELF_TEST_FAILURES + 1))
    fi
}

summary_line() {
    echo "test result: $1. $2 passed; $3 failed; $4 ignored; 0 measured; $5 filtered out; finished in 0.01s"
}

write_target_case() {
    local dir="$WORK_DIR/case-$1"
    mkdir -p "$dir"
    printf 'alpha\nbeta\ngamma_panics\n' >"$dir/expected"
    echo "$dir"
}

self_test_targets() {
    local d
    d="$(write_target_case pass)"
    printf 'alpha: test\nbeta: test\ngamma_panics: test\n' >"$d/list"
    { echo "running 3 tests"; echo "test alpha ... ok"; echo "test gamma_panics - should panic ... ok"
      echo "test beta ... ok"; echo; summary_line ok 3 0 0 0; } >"$d/run"
    expect pass "matching inventory with a should-panic test" check_target t "$d/expected" "$d/list" "$d/run" "$d"

    d="$(write_target_case missing)"
    printf 'alpha: test\nbeta: test\n' >"$d/list"
    { echo "test alpha ... ok"; echo "test beta ... ok"; summary_line ok 2 0 0 0; } >"$d/run"
    expect fail "missing test" check_target t "$d/expected" "$d/list" "$d/run" "$d"

    d="$(write_target_case unlisted)"
    printf 'alpha: test\nbeta: test\ndelta: test\ngamma_panics: test\n' >"$d/list"
    { echo "test alpha ... ok"; echo "test beta ... ok"; echo "test delta ... ok"
      echo "test gamma_panics - should panic ... ok"; summary_line ok 4 0 0 0; } >"$d/run"
    expect fail "unlisted test" check_target t "$d/expected" "$d/list" "$d/run" "$d"

    d="$(write_target_case ignored)"
    printf 'alpha: test\nbeta: test\ngamma_panics: test\n' >"$d/list"
    { echo "test alpha ... ok"; echo "test beta ... ignored"
      echo "test gamma_panics - should panic ... ok"; summary_line ok 2 0 1 0; } >"$d/run"
    expect fail "ignored test" check_target t "$d/expected" "$d/list" "$d/run" "$d"

    d="$(write_target_case failed)"
    printf 'alpha: test\nbeta: test\ngamma_panics: test\n' >"$d/list"
    { echo "test alpha ... ok"; echo "test beta ... FAILED"
      echo "test gamma_panics - should panic ... ok"; summary_line FAILED 2 1 0 0; } >"$d/run"
    expect fail "failed test" check_target t "$d/expected" "$d/list" "$d/run" "$d"

    d="$(write_target_case filtered)"
    printf 'alpha: test\nbeta: test\ngamma_panics: test\n' >"$d/list"
    { echo "test alpha ... ok"; echo "test beta ... ok"; summary_line ok 2 0 0 1; } >"$d/run"
    expect fail "filtered-out test" check_target t "$d/expected" "$d/list" "$d/run" "$d"

    d="$(write_target_case zero)"
    : >"$d/list"
    { echo "running 0 tests"; echo; summary_line ok 0 0 0 0; } >"$d/run"
    expect fail "zero executed tests" check_target t "$d/expected" "$d/list" "$d/run" "$d"

    d="$(write_target_case two-summaries)"
    printf 'alpha: test\nbeta: test\ngamma_panics: test\n' >"$d/list"
    { echo "test alpha ... ok"; echo "test beta ... ok"; echo "test gamma_panics - should panic ... ok"
      summary_line ok 3 0 0 0; summary_line ok 3 0 0 0; } >"$d/run"
    expect fail "two libtest summaries" check_target t "$d/expected" "$d/list" "$d/run" "$d"

    d="$(write_target_case summary-filtered)"
    printf 'alpha: test\nbeta: test\ngamma_panics: test\n' >"$d/list"
    { echo "test alpha ... ok"; echo "test beta ... ok"; echo "test gamma_panics - should panic ... ok"
      summary_line ok 3 0 0 1; } >"$d/run"
    expect fail "summary reports a filtered-out test" check_target t "$d/expected" "$d/list" "$d/run" "$d"

    d="$(write_target_case summary-count)"
    printf 'alpha: test\nbeta: test\ngamma_panics: test\n' >"$d/list"
    { echo "test alpha ... ok"; echo "test beta ... ok"; echo "test gamma_panics - should panic ... ok"
      summary_line ok 4 0 0 0; } >"$d/run"
    expect fail "summary count differs from the manifest" check_target t "$d/expected" "$d/list" "$d/run" "$d"

    d="$(write_target_case panic-failed)"
    printf 'alpha: test\nbeta: test\ngamma_panics: test\n' >"$d/list"
    { echo "test alpha ... ok"; echo "test beta ... ok"
      echo "test gamma_panics - should panic ... FAILED"; summary_line FAILED 2 1 0 0; } >"$d/run"
    expect fail "should-panic test that did not panic" check_target t "$d/expected" "$d/list" "$d/run" "$d"
}

write_manifest() {
    local path="$WORK_DIR/manifest-$1.txt"
    cat >"$path"
    echo "$path"
}

valid_manifest_body() {
    echo "# comment line"
    echo
    echo "thresh-association mht_reference c1_enumerates_27_hypotheses"
    echo "thresh-filter imm_reference asymmetric_mixing_matches_exact_rationals  # trailing comment"
    echo "thresh-filter imm_current_policy current_policy_default_floor"
}

self_test_manifests() {
    local m
    m="$(valid_manifest_body | write_manifest valid)"
    expect pass "valid manifest" validate_manifest "$m" "$WORK_DIR/out-valid"

    m="$(printf '# only comments\n\n' | write_manifest empty)"
    expect fail "empty manifest" validate_manifest "$m" "$WORK_DIR/out-empty"

    m="$({ valid_manifest_body; echo "thresh-filter imm_reference asymmetric_mixing_matches_exact_rationals"; } | write_manifest duplicated)"
    expect fail "duplicated manifest entry" validate_manifest "$m" "$WORK_DIR/out-dup"

    m="$({ valid_manifest_body; echo "thresh-filter imm_reference"; } | write_manifest two-fields)"
    expect fail "malformed manifest entry (two fields)" validate_manifest "$m" "$WORK_DIR/out-two"

    m="$({ valid_manifest_body; echo "thresh-filter imm_reference a b"; } | write_manifest four-fields)"
    expect fail "malformed manifest entry (four fields)" validate_manifest "$m" "$WORK_DIR/out-four"

    m="$({ valid_manifest_body; echo "thresh-filter imm_reference bad-name"; } | write_manifest bad-chars)"
    expect fail "malformed manifest entry (invalid test name)" validate_manifest "$m" "$WORK_DIR/out-bad"

    m="$(valid_manifest_body | grep -v imm_current_policy | write_manifest no-required)"
    expect fail "required target without entries" validate_manifest "$m" "$WORK_DIR/out-noreq"

    expect fail "missing manifest file" validate_manifest "$WORK_DIR/does-not-exist.txt" "$WORK_DIR/out-none"
}

run_self_test() {
    make_work_dir
    self_test_targets
    self_test_manifests
    if [ "$SELF_TEST_FAILURES" -ne 0 ]; then
        echo "Native reference gate self-test FAILED: $SELF_TEST_FAILURES of $SELF_TEST_CASES cases" >&2
        return 1
    fi
    echo "Native reference gate self-test PASSED: $SELF_TEST_CASES cases."
}

usage() {
    echo "usage: $0 [--manifest PATH] | --self-test" >&2
}

main() {
    local manifest="$DEFAULT_MANIFEST"
    case "${1:-}" in
        --self-test)
            [ "$#" -eq 1 ] || { usage; return 2; }
            run_self_test
            return
            ;;
        --manifest)
            [ "$#" -eq 2 ] || { usage; return 2; }
            manifest="$2"
            ;;
        "") ;;
        *)
            usage
            return 2
            ;;
    esac
    run_gate "$manifest"
}

main "$@"
