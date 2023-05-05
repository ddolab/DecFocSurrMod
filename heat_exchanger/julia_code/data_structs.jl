mutable struct IOPVars
    primal::AbstractArray
    dual::AbstractArray
    slacks::AbstractArray
    obj_params::AbstractArray
    constraint_params::AbstractArray
end

mutable struct AlgParams
    η::Float64
    γ::Float64
    penalty_params::AbstractArray
    max_outer_iter::Int64
    max_inner_iter::Int64
end
