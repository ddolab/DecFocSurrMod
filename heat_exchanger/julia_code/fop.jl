function forward_problem(T5)
    T3 = 388
    FP = Model(
        optimizer_with_attributes(
            () -> Gurobi.Optimizer(gurobi_env),
            "OutputFlag" => 0,
            "NonConvex" => 2,
        ),
    )
    JuMP.@variables(FP, begin
        Qc >= 0
        FH >= 0
    end)

    @constraint(FP, Qc / 2 + 553 - T3 >= 0)
    @constraint(FP, -10 - Qc + (T5 - T3 - 170 + 0.5 * Qc) * FH >= 0)
    @constraint(FP, 2 * T3 - 786 - Qc + (T5 - 393) * FH >= 0)
    @constraint(FP, 2 * T3 - 1026 - Qc + (T5 - 313) * FH >= 0)
    @constraint(FP, 2 * T3 - 1026 - Qc + (T5 - 323) * FH <= 0)
    @objective(FP, Min, 1e-2 * Qc + 4 * (FH - 1.7)^2)
    optimize!(FP)
    return [value.(Qc), value.(FH)]
end

function surrogate_fop(p, q, T5)
    SFP = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    T3 = 388
    a = sum(p[1, j] * (T5)^(j - 1) for j = 1:degree)
    b = sum(p[2, j] * (T5)^(j - 1) for j = 1:degree)
    Q = [sum(q[i, j] * (T5)^(j - 1) for j = 1:degree) for i = 1:4]

    JuMP.@variables(SFP, begin
        Qc >= 0, (start = 200)
        FH >= 0
    end)

    @constraint(SFP, Qc / 2 + 553 - T3 >= 0)
    @constraint(SFP, -10 - Qc + (T5 - T3 - 170) * FH + a * Qc + b * FH >= 0)
    @constraint(SFP, 2 * T3 - 786 - Qc + (T5 - 393) * FH >= 0)
    @constraint(SFP, 2 * T3 - 1026 - Qc + (T5 - 313) * FH >= 0)
    @constraint(SFP, 2 * T3 - 1026 - Qc + (T5 - 323) * FH <= 0)
    @objective(SFP, Min, Q[1] * Qc^2 + Q[2] * Qc + Q[3] * FH^2 + Q[4] * FH)
    optimize!(SFP)
    return [value.(Qc), value.(FH)]
end

function SurrogateBoundingProblem(Q, p, T5)
    T3 = 388
    SFP = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    a = sum(p[1, j] * (T5)^(j - 1) for j = 1:degree)
    b = sum(p[2, j] * (T5)^(j - 1) for j = 1:degree)

    JuMP.@variables(SFP, begin
        Qc >= 0
        FH >= 0
    end)

    @constraint(SFP, Qc / 2 + 553 - T3 >= 0)
    @constraint(SFP, -10 - Qc + (T5 - T3 - 170) * FH + a * Qc + b * FH >= 0)
    @constraint(SFP, 2 * T3 - 786 - Qc + (T5 - 393) * FH >= 0)
    @constraint(SFP, 2 * T3 - 1026 - Qc + (T5 - 313) * FH >= 0)
    @constraint(SFP, 2 * T3 - 1026 - Qc + (T5 - 323) * FH <= 0)
    @objective(SFP, Min, Qc)
    optimize!(SFP)
    lb = objective_value(SFP)
    @objective(SFP, Max, Qc)
    optimize!(SFP)
    ub = objective_value(SFP)
    return lb, ub
end
