using Printf

const DEFAULT_RESOLUTIONS = [129, 257, 513]
const DEFAULT_TESTCASES = [
	"Diagonal",
	"Stretch",
	"Rotate",
	"CircularVortex",
	"ReversedVortex",
]

function parse_csv_strings(value::String)
	items = split(value, ",")
	return [strip(item) for item in items if !isempty(strip(item))]
end

function parse_csv_ints(value::String)
	values = parse_csv_strings(value)
	return [parse(Int, v) for v in values]
end

function env_or_default_strings(varname::String, default::Vector{String})
	raw = get(ENV, varname, "")
	isempty(strip(raw)) && return default
	parsed = parse_csv_strings(raw)
	isempty(parsed) && return default
	return parsed
end

function env_or_default_ints(varname::String, default::Vector{Int})
	raw = get(ENV, varname, "")
	isempty(strip(raw)) && return default
	parsed = parse_csv_ints(raw)
	isempty(parsed) && return default
	return parsed
end

function run_single_case(repo_root::String, tracer_script::String, n::Int, case_name::String)
	println("============================================================")
	println("Running tracer case=$(case_name), n=$(n)")

	cmd = `$(Base.julia_cmd()) --project=$(repo_root) $(tracer_script)`
	start_time = time()

	ok = true
	err_msg = ""

	withenv("CS_TRACER_N" => string(n), "CS_TRACER_SIM" => case_name) do
		try
			run(cmd)
		catch err
			ok = false
			err_msg = sprint(showerror, err)
		end
	end

	elapsed = time() - start_time

	if ok
		println(@sprintf("Finished case=%s, n=%d in %.2fs", case_name, n, elapsed))
	else
		println(@sprintf("FAILED case=%s, n=%d in %.2fs", case_name, n, elapsed))
		println("Error: $(err_msg)")
	end

	flush(stdout)
	return (n = n, case = case_name, ok = ok, seconds = elapsed, error = err_msg)
end

function print_summary(results)
	println("\n======================== Tracer Sweep Summary ========================")

	total = length(results)
	passed = count(r -> r.ok, results)
	failed = total - passed

	for r in results
		status = r.ok ? "PASS" : "FAIL"
		println(@sprintf("[%s] n=%4d | case=%-16s | %8.2fs", status, r.n, r.case, r.seconds))
	end

	println("---------------------------------------------------------------------")
	println(@sprintf("Total: %d | Passed: %d | Failed: %d", total, passed, failed))
end

function main()
	repo_root = normpath(joinpath(@__DIR__, "..", ".."))
	tracer_script = joinpath(@__DIR__, "Tracer.jl")

	resolutions = env_or_default_ints("CS_TRACER_RESOLUTIONS", DEFAULT_RESOLUTIONS)
	testcases = env_or_default_strings("CS_TRACER_TESTCASES", DEFAULT_TESTCASES)

	println("Tracer sweep starting...")
	println("Resolutions: $(resolutions)")
	println("Testcases: $(testcases)")

	results = NamedTuple[]
	for n in resolutions
		for case_name in testcases
			push!(results, run_single_case(repo_root, tracer_script, n, case_name))
		end
	end

	print_summary(results)

	if any(!r.ok for r in results)
		error("One or more tracer sweep runs failed.")
	end

	return nothing
end

main()
