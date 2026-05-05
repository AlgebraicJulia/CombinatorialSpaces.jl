abstract type AbstractSimulationModel end

struct LMNSModel <: AbstractSimulationModel end
struct LMNSHModel <: AbstractSimulationModel end
struct MHDModel <: AbstractSimulationModel end

Base.@kwdef struct CallbackConfig{FT <: AbstractFloat}
  te::FT
  dt::FT
  saveat::Int = 500
  checkpoint_at::Int = 10_000
  start_step::Int = 1
  start_time::FT = zero(FT)
end

step_from_iter(cfg::CallbackConfig, iter::Int) = cfg.start_step - 1 + iter

# Required model-dispatched hooks.
build_saved_value_type(::AbstractSimulationModel, ::Type{FT}) where FT <: AbstractFloat =
  error("build_saved_value_type is not implemented for this model")

regularized_state(::AbstractSimulationModel, u, context) =
  error("regularized_state is not implemented for this model")

# Optional model-dispatched hooks.
progress_reference(::AbstractSimulationModel, u0, context) = (;)
log_progress!(::AbstractSimulationModel, integrator, refs, cfg::CallbackConfig, context) = nothing

checkpoint_schema_version(::AbstractSimulationModel) = 1
checkpoint_model_kind(::AbstractSimulationModel) = error("checkpoint_model_kind is not implemented for this model")
checkpoint_field_names(::AbstractSimulationModel) = error("checkpoint_field_names is not implemented for this model")

function checkpoint_to_device(context)
  if context !== nothing && hasproperty(context, :to_device)
    return getproperty(context, :to_device)
  end
  return identity
end

function _checkpoint_tail_state(raw_state)
  if raw_state isa AbstractVector && !isempty(raw_state)
    return raw_state[end]
  end
  if hasproperty(raw_state, :u)
    state_hist = getproperty(raw_state, :u)
    if state_hist isa AbstractVector && !isempty(state_hist)
      return state_hist[end]
    end
  end
  return raw_state
end

function _split_flat_state(state_flat::AbstractVector, lengths::Vector{Int})
  values = Vector{Any}(undef, length(lengths))
  first_idx = 1
  for i in eachindex(lengths)
    last_idx = first_idx + lengths[i] - 1
    values[i] = state_flat[first_idx:last_idx]
    first_idx = last_idx + 1
  end
  return values
end

function pack_checkpoint_state(model::AbstractSimulationModel, u, context)
  fields = checkpoint_field_names(model)
  return Tuple(Array(getproperty(u, field)) for field in fields)
end

function unpack_checkpoint_state(model::AbstractSimulationModel, raw_state, template_state, context)
  fields = checkpoint_field_names(model)
  state_like = _checkpoint_tail_state(raw_state)
  values = nothing

  if all(field -> hasproperty(state_like, field), fields)
    values = [getproperty(state_like, field) for field in fields]
  elseif state_like isa Tuple && length(state_like) == length(fields)
    values = collect(state_like)
  elseif state_like isa AbstractVector
    lengths = [length(getproperty(template_state, field)) for field in fields]
    if length(state_like) != sum(lengths)
      throw(ArgumentError("Unsupported flat checkpoint_state length for model $(checkpoint_model_kind(model))"))
    end
    values = _split_flat_state(state_like, lengths)
  else
    throw(ArgumentError("Unsupported checkpoint_state format for model $(checkpoint_model_kind(model))"))
  end

  to_device = checkpoint_to_device(context)
  device_values = map(value -> to_device(Array(value)), values)
  state_nt = NamedTuple{fields}(Tuple(device_values))
  return ComponentArray(state_nt)
end

function latest_checkpoint_path(checkpoint_dir::AbstractString)
  isdir(checkpoint_dir) || return nothing

  re = r"^checkpoint_step_(\d+)\.jld2$"
  latest_step = -1
  latest_file = nothing

  for file in readdir(checkpoint_dir)
    m = match(re, file)
    m === nothing && continue

    step = parse(Int, m.captures[1])
    if step > latest_step
      latest_step = step
      latest_file = joinpath(checkpoint_dir, file)
    end
  end

  return latest_file
end

function load_checkpoint_state(model::AbstractSimulationModel, checkpoint_path::AbstractString, template_state; context=nothing)
  data = JLD2.load(checkpoint_path)

  if haskey(data, "checkpoint_schema_version")
    saved_schema = Int(data["checkpoint_schema_version"])
    expected_schema = checkpoint_schema_version(model)
    if saved_schema > expected_schema
      throw(ArgumentError("Checkpoint schema version $(saved_schema) is newer than supported version $(expected_schema)."))
    end
  end

  if haskey(data, "model_kind")
    saved_kind_raw = data["model_kind"]
    saved_kind = saved_kind_raw isa Symbol ? saved_kind_raw : Symbol(saved_kind_raw)
    expected_kind = checkpoint_model_kind(model)
    if saved_kind != expected_kind
      throw(ArgumentError("Checkpoint model kind $(saved_kind) does not match requested model $(expected_kind)."))
    end
  end

  if !haskey(data, "checkpoint_state")
    return nothing
  end

  checkpoint_t = Float64(data["checkpoint_t"])
  checkpoint_step = Int(data["checkpoint_step"])
  checkpoint_state = data["checkpoint_state"]

  return (
    state = unpack_checkpoint_state(model, checkpoint_state, template_state, context),
    checkpoint_t = checkpoint_t,
    checkpoint_step = checkpoint_step,
  )
end

function initialize_or_resume_state(model::AbstractSimulationModel, initial_state; resume::Bool=false, checkpoint_dir::AbstractString, context=nothing)
  if resume
    checkpoint_path = latest_checkpoint_path(checkpoint_dir)
    if checkpoint_path !== nothing
      loaded = load_checkpoint_state(model, checkpoint_path, initial_state; context=context)
      if loaded !== nothing
        start_step = loaded.checkpoint_step + 1
        start_time = loaded.checkpoint_t
        println("Resuming from checkpoint $(checkpoint_path)")
        println("Resuming from t=$(start_time), step=$(start_step)")
        return (
          state = loaded.state,
          start_time = start_time,
          start_step = start_step,
          resumed = true,
          checkpoint_path = checkpoint_path,
        )
      end
      println("Checkpoint $(checkpoint_path) does not contain checkpoint_state; starting from initial condition.")
    else
      println("No checkpoint found in $(checkpoint_dir); starting from initial condition.")
    end
  end

  return (
    state = deepcopy(initial_state),
    start_time = 0.0,
    start_step = 1,
    resumed = false,
    checkpoint_path = nothing,
  )
end

model_has_smoothing(::AbstractSimulationModel) = false
apply_smoothing!(::AbstractSimulationModel, integrator, context) = nothing

model_has_periodic_prestep(::AbstractSimulationModel, context) = false
apply_periodic_prestep!(::AbstractSimulationModel, integrator, context) = nothing

run_checkpoint_outputs!(::AbstractSimulationModel, regular_save_values::SavedValues, step::Int, checkpoint_t::AbstractFloat, cfg::CallbackConfig, context) = nothing

function periodic_side_selection(full_periodic::Bool, periodic_left_right::Bool, periodic_top_bottom::Bool)
  full_periodic && return ALL
  periodic_left_right && periodic_top_bottom && return ALL
  periodic_left_right && return EASTWEST
  periodic_top_bottom && return NORTHSOUTH
  return nothing
end

function run_with_model_callbacks(
  model::AbstractSimulationModel,
  u0,
  rhs!,
  params,
  cfg::CallbackConfig{FT};
  context = nothing,
) where FT <: AbstractFloat

  refs = progress_reference(model, u0, context)

  regular_save_values = SavedValues(
    FT,
    build_saved_value_type(model, FT),
  )

  regular_state_cb = SavingCallback(
    (u, t, integrator) -> regularized_state(model, u, context),
    regular_save_values;
    saveat = cfg.saveat * cfg.dt,
    save_start = true,
    save_end = false,
  )

  function smoothing_condition(u, t, integrator)
    integrator.iter > 0 || return false
    return model_has_smoothing(model)
  end

  function smoothing_affect!(integrator)
    apply_smoothing!(model, integrator, context)
    return nothing
  end

  function save_condition(u, t, integrator)
    integrator.iter > 0 || return false
    step = step_from_iter(cfg, integrator.iter)
    return step % cfg.saveat == 0
  end

  function save_affect!(integrator)
    log_progress!(model, integrator, refs, cfg, context)
    return nothing
  end

  function checkpoint_condition(u, t, integrator)
    integrator.iter > 0 || return false
    step = step_from_iter(cfg, integrator.iter)
    return step % cfg.checkpoint_at == 0
  end

  function checkpoint_affect!(integrator)
    step = step_from_iter(cfg, integrator.iter)

    schema_version = checkpoint_schema_version(model)
    model_kind = checkpoint_model_kind(model)
    field_names = checkpoint_field_names(model)
    checkpoint_t = integrator.t
    checkpoint_step = step
    checkpoint_state = pack_checkpoint_state(model, integrator.u, context)
    checkpoint_regular_t = regular_save_values.t
    checkpoint_regular_state = regular_save_values.saveval

    @save joinpath(context.save_path, "checkpoint_step_$(step).jld2") schema_version model_kind field_names checkpoint_t checkpoint_step checkpoint_state checkpoint_regular_t checkpoint_regular_state

    if isempty(checkpoint_regular_state)
      push!(regular_save_values.t, checkpoint_t)
      push!(regular_save_values.saveval, regularized_state(model, integrator.u, context))
    end

    run_checkpoint_outputs!(model, regular_save_values, step, checkpoint_t, cfg, context)

    # Keep continuity at checkpoint boundaries while limiting memory growth.
    empty!(regular_save_values.t)
    empty!(regular_save_values.saveval)
    push!(regular_save_values.t, checkpoint_t)
    push!(regular_save_values.saveval, regularized_state(model, integrator.u, context))

    return nothing
  end

  callback_items = Any[]
  if model_has_periodic_prestep(model, context)
    function periodic_prestep_affect!(integrator)
      apply_periodic_prestep!(model, integrator, context)
      return nothing
    end

    push!(callback_items, PeriodicCallback(periodic_prestep_affect!, cfg.dt; initial_affect = true, final_affect = false))
  end

  if model_has_smoothing(model)
    push!(callback_items, DiscreteCallback(smoothing_condition, smoothing_affect!; save_positions = (false, false)))
  end
  push!(callback_items, regular_state_cb)
  push!(callback_items, DiscreteCallback(save_condition, save_affect!; save_positions = (false, false)))
  push!(callback_items, DiscreteCallback(checkpoint_condition, checkpoint_affect!; save_positions = (false, false)))

  callbacks = CallbackSet(callback_items...)

  prob = ODEProblem(rhs!, u0, (cfg.start_time, cfg.te), params)
  solve(
    prob,
    SSPRK33();
    dt = cfg.dt,
    adaptive = false,
    save_everystep = false,
    save_start = false,
    save_end = false,
    callback = callbacks,
    dense = false,
  )

  return nothing
end
