using Printf
using Dates

const DEFAULT_RESOLUTIONS = [129, 257, 513, 1025]
const DEFAULT_TESTCASES = [
	"Diagonal",
	"Stretch",
	"Rotate",
	"CircularVortex",
	# "ReversedVortex", # TODO: This case is currently failing, needs investigation
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

function run_single_case(repo_root::String, tracer_script::String, n::Int, case_name::String, log_io::IO)
	write(log_io, "============================================================\n")
	write(log_io, "Running tracer case=$(case_name), n=$(n)\n")

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
		write(log_io, @sprintf("Finished case=%s, n=%d in %.2fs\n", case_name, n, elapsed))
	else
		write(log_io, @sprintf("FAILED case=%s, n=%d in %.2fs\n", case_name, n, elapsed))
		write(log_io, "Error: $(err_msg)\n")
	end

	flush(log_io)
	return (n = n, case = case_name, ok = ok, seconds = elapsed, error = err_msg)
end

function print_summary(results, log_io::IO)
	write(log_io, "\n======================== Tracer Sweep Summary ========================\n")

	total = length(results)
	passed = count(r -> r.ok, results)
	failed = total - passed

	for r in results
		status = r.ok ? "PASS" : "FAIL"
		write(log_io, @sprintf("[%s] n=%4d | case=%-16s | %8.2fs\n", status, r.n, r.case, r.seconds))
	end

	write(log_io, "---------------------------------------------------------------------\n")
	write(log_io, @sprintf("Total: %d | Passed: %d | Failed: %d\n", total, passed, failed))
end

function main()
	repo_root     = normpath(joinpath(@__DIR__, "..", ".."))
	tracer_script = joinpath(@__DIR__, "Tracer.jl")

	rm(joinpath(@__DIR__, "imgs", "Tracer"); force = true, recursive = true) # Clear old images

	resolutions = env_or_default_ints("CS_TRACER_RESOLUTIONS", DEFAULT_RESOLUTIONS)
	testcases   = env_or_default_strings("CS_TRACER_TESTCASES", DEFAULT_TESTCASES)

	timestamp = Dates.format(now(), "yyyy-mm-ddTHH-MM-SS")
	log_path  = joinpath(@__DIR__, "tracer_results_$(timestamp).txt")

	open(log_path, "w") do log_io
		write(log_io, "Tracer sweep starting...\n")
		write(log_io, "Resolutions: $(resolutions)\n")
		write(log_io, "Testcases: $(testcases)\n")

		results = NamedTuple[]
		for n in resolutions
			for case_name in testcases
				push!(results, run_single_case(repo_root, tracer_script, n, case_name, log_io))
			end
		end

		print_summary(results, log_io)
		flush(log_io)

		if any(!r.ok for r in results)
			error("One or more tracer sweep runs failed.")
		end
	end

	return nothing
end

main()
