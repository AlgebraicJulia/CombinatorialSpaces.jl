module Tonti
using Catlab.Present
using Catlab.Theories
using Catlab.CategoricalAlgebra
using CombinatorialSpaces
export TontiDiagram, AbstractTontiDiagram, addTransform!, addSpace!,
       addTime!, vectorfield

@present TheoryTontiDiagram(FreeSchema) begin
  Func::Data
  Label::Data

  Variable::Ob
  Corner::Ob
  Transform::Ob
  InParam::Ob
  OutParam::Ob

  tgt::Hom(InParam, Transform)
  src::Hom(OutParam, Transform)
  in_var::Hom(InParam, Variable)
  out_var::Hom(OutParam, Variable)
  
  func::Attr(Transform, Func)
  t_type::Attr(Transform, Label)

  corner::Hom(Variable, Corner)

  c_label::Attr(Corner, Label)
  v_label::Attr(Variable, Label)
end

const AbstractTontiDiagram = AbstractACSetType(TheoryTontiDiagram)
const TontiDiagram = ACSetType(TheoryTontiDiagram,index=[:src, :tgt, :corner,
                                                         :in_var, :out_var],
                                                 unique_index=[:v_label,
                                                               :c_label])

function TontiDiagram(dimension = 3)
  td = TontiDiagram{Function, Symbol}()
  corner_syms = [:P, :L, :S, :V]

  for d in 1:(dimension+1)
    d2 = dimension - (d-1) + 1
    primal = add_parts!(td, :Corner, 2, c_label=[Symbol("I$(corner_syms[d])"),
                                                  Symbol("T$(corner_syms[d])")])
    dual = add_parts!(td, :Corner, 2, c_label=[Symbol("I$(corner_syms[d2])2"),
                                                Symbol("T$(corner_syms[d2])2")])
  end
  td
end

function TontiDiagram(dimension::Int, variables::Array{Pair{Symbol, Symbol}})
  td = TontiDiagram(dimension)
  add_parts!(td, :Variable, length(variables), v_label=first.(variables),
             corner=[first(incident(td, s, :c_label)) for s in last.(variables)])
  td
end

dimension(td::AbstractTontiDiagram) = nparts(td, :Corner) ÷ 4 - 1

var_corner(td::AbstractTontiDiagram, label::Symbol) = td[incident(td, :f, :v_label), :corner]
get_var_ind(td::AbstractTontiDiagram, label::Symbol) = first(incident(td, label, :v_label))

function addTransform!(td::AbstractTontiDiagram, dom::Array{Symbol}, func::Function, codom::Array{Symbol})
  # Check rules for transforms
  dom_sim = var_corner(td, first(dom))
  codom_sim = var_corner(td, first(codom))
  all(var_corner(td, d) == dom_sim for d in dom) || error("Domain is not consistently on the corner $dom_sim")
  all(var_corner(td, c) == codom_sim for c in codom) || error("Codomain is not consistently on the corner $codom_sim")

  # Add transformation
  tran = add_part!(td, :Transform, func=func, t_type=:constitutive)
  add_parts!(td, :InParam, length(dom), tgt=tran, in_var=[get_var_ind(td, v) for v in dom])
  add_parts!(td, :OutParam, length(codom), src=tran, out_var=[get_var_ind(td, v) for v in codom])
end

function addTopoTransform!(td::AbstractTontiDiagram, dom::Symbol, func::Function, codom::Symbol)
  v_dom = incident(td, dom, [:corner, :c_label])
  v_codom = incident(td, codom, [:corner, :c_label])
  
  (length(v_dom) <= 1 && length(v_codom) <= 1) || 
      error("Adding topological transformations with >1 variable per corner is not yet supported")

  if length(v_dom) == 0 || length(v_codom) == 0
    return false
  end

  tran = add_part!(td, :Transform, func=func, t_type=:topological)
  add_part!(td, :InParam, tgt=tran, in_var=v_dom[1])
  add_part!(td, :OutParam, src=tran, out_var=v_codom[1])
  return true
end

function addTempTransform!(td::AbstractTontiDiagram, dom::Symbol, func::Function, codom::Symbol)
  v_dom = incident(td, dom, [:corner, :c_label])
  v_codom = incident(td, codom, [:corner, :c_label])
  
  (length(v_dom) <= 1 && length(v_codom) <= 1) || 
      error("Adding topological transformations with >1 variable per corner is not yet supported")

  if length(v_dom) == 0 || length(v_codom) == 0
    return false
  end

  tran = add_part!(td, :Transform, func=func, t_type=:temporal)
  add_part!(td, :InParam, tgt=tran, in_var=v_dom[1])
  add_part!(td, :OutParam, src=tran, out_var=v_codom[1])
  return true
end

function addSpace!(td::AbstractTontiDiagram, complex::AbstractSemiSimplicialSet1D)
  bound_1_0   = boundary(1,complex)
  cobound_0_1 = d(0,complex)
  addTopoTransform!(td, :IP, x->(cobound_0_1*x), :IL)
  addTopoTransform!(td, :TP, x->(cobound_0_1*x), :TL)
 
  # TODO: Add Hodge * operator instead of just swapping bound/cobound 
  addTopoTransform!(td, :TP, x->(bound_1_0*x), :TL)
  addTopoTransform!(td, :TP2, x->(bound_1_0*x), :TL2)

  td
end



function addSpace!(td::AbstractTontiDiagram, complex::AbstractSemiSimplicialSet2D)

  td
end

function addTime!(td::AbstractTontiDiagram; dt=1)
  addTempTransform!(td, :TP, x->dt*x, :IP)
  addTempTransform!(td, :TL, x->dt*x, :IL)
  addTempTransform!(td, :TP2, x->dt*x, :IP2)
  addTempTransform!(td, :TL2, x->dt*x, :IL2)
  
  if dimension(td) > 1
    addTempTransform!(td, :TS, x->dt*x, :IS)
    addTempTransform!(td, :TS2, x->dt*x, :IS2)
  end
  
  if dimension(td) > 2
    addTempTransform!(td, :TV, x->dt*x, :IV)
    addTempTransform!(td, :TV2, x->dt*x, :IV2)
  end
end

function vectorfield(td::AbstractTontiDiagram, complex::AbstractSemiSimplicialSet1D)
  # Define order of evaluation for transformations
  var_deps = [Set(filter!(t -> (td[t, :t_type] != :temporal), incident(td, i, :out_var)))
              for i in 1:nparts(td, :Variable)] # deps per variable

  tran_deps = [Set(td[incident(td, t, :src), :in_var])
               for t in 1:nparts(td, :Transform)] # deps per transforms

  evaluated = fill(false, nparts(td, :Transform))
  to_evaluate = Array{Int64, 1}()
  to_eval_vars = Array{Int64, 1}()
  order = Array{Int64, 1}()
  time_vars = td[incident(td, :temporal, [:src, :t_type]), :out_var]

  append!(to_eval_vars, time_vars)
  # Add any transformations which now have all variables satisfied
  for v in to_eval_vars
    for t in 1:length(tran_deps)
      delete!(tran_deps[t], v)
      if isempty(tran_deps[t]) && !evaluated[t]
        push!(to_evaluate, t)
        evaluated[t] = true
      end
    end
  end
  to_eval_vars = empty(to_eval_vars)

  # Evaluate transforms which have no more unsatisfied variable dependencies
  while !isempty(to_evaluate)
    cur_tran = pop!(to_evaluate)
    push!(order, cur_tran)
    
    outputs = incident(td, cur_tran, :src)
    out_vars = td[outputs, :out_var]
    
    # Update variable dependencies based on this transformation's eval
    for out in 1:length(outputs)
      delete!(var_deps[out_vars[out]], outputs[out])
      if isempty(var_deps[out_vars[out]])
        push!(to_eval_vars, out_vars[out])
      end
    end

    # Add transforms which have all transforms satisfied
    for v in to_eval_vars
      for t in 1:length(tran_deps)
        delete!(tran_deps[t], v)
        if isempty(tran_deps[t]) && !evaluated[t]
          push!(to_evaluate, t)
          evaluated[t] = true
        end
      end
    end
    to_eval_vars = empty(to_eval_vars)
  end

  # Clear out "temporal" transformations
  # filter!(t -> (td[t, :t_type] != :temporal), order)

  # Initialize memory
  data = initData(td, complex)
  timevar_to_ind = Dict{Int, Int}()
  for v in 1:length(time_vars)
    push!(data, copy(data[time_vars[v]]))
    timevar_to_ind[time_vars[v]] = length(data)
  end

  transforms = Array{Pair{Array{Int,1},Array{Int,1}},1}()
  for t in 1:nparts(td, :Transform)
    in_vars = [(x ∈ time_vars) ? timevar_to_ind[x] : x for x in td[incident(td, t, :tgt), :in_var]]
    out_vars = td[incident(td, t, :src), :out_var]
    push!(transforms, in_vars => out_vars)
  end

  function system(du, u, t, p)
    # Reset data to 0
    for i in 1:length(data)
      data[i] .= 0
    end

    # Update current state from u
    cur_ind = 1
    for i in 1:length(time_vars)
      v_len = length(data[timevar_to_ind[time_vars[i]]])
      data[timevar_to_ind[time_vars[i]]] .= u[cur_ind:(cur_ind+v_len-1)]
      cur_ind += v_len
    end

    for t in order
      if length(transforms[t][2]) == 1
        data[transforms[t][2]][1] .+= subpart(td, t, :func)(data[transforms[t][1]]...)
      else
        data[transforms[t][2]] .+= subpart(td, t, :func)(data[transforms[t][1]]...)
      end
    end
    du .= vcat(data[time_vars]...)
  end
  system, length.(data[time_vars])
end

function initData(td::AbstractTontiDiagram, complex::AbstractSemiSimplicialSet1D)
  data = Array{Array{Float64, 1}, 1}()
  corner_to_len = Dict(:IP => nv(complex), :TP => nv(complex), :IL2 => nv(complex), :TL2 => nv(complex),
                       :IL => ne(complex), :TL => ne(complex), :IP2 => ne(complex), :TP2 => ne(complex))

  for v in 1:nparts(td, :Variable)
    push!(data, zeros(corner_to_len[td[td[v, :corner], :c_label]]))
  end
  data
end
end
