function inverse_problem()
    T3 = 388
    IOP = Model(
        optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0, "max_iter" => 10000),
    )

    # IOP = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env), "OutputFlag" => 1, 
    # "NonConvex" => 2))
    JuMP.@variables(IOP, begin
        λ[i = 1:NoS, j = 1:5] >= 0
        hat_Qc[i = 1:NoS], (start = x[i][1])
        hat_FH[i = 1:NoS], (start = x[i][2])
        a[1:NoS]
        b[1:NoS]
        p[i = 1:2, j = 1:degree]
        q[i = 1:4, j = 1:degree]
        Q[1:NoS, 1:4]
        0 <= t[i = 1:2, j = 1:degree]
        0 <= tt[1:4, 1:degree]
        dual1[1:2, 1:NoS] >= 0
        dual2[1:5, 1:NoS] >= 0
        dual3[1:NoS] <= 0
        w1[1:NoS]
        w2[1:NoS]
    end)
    @constraint(
        IOP,
        [s = 1:NoS],
        2 * Q[s, 1] * hat_Qc[s] + Q[s, 2] - 0.5 * λ[s, 1] - (a[s] - 1) * λ[s, 2] +
        λ[s, 3] +
        λ[s, 4] - λ[s, 5] <= dual1[1, s]
    )
    @constraint(
        IOP,
        [s = 1:NoS],
        -(
            2 * Q[s, 1] * hat_Qc[s] + Q[s, 2] - 0.5 * λ[s, 1] - (a[s] - 1) * λ[s, 2] +
            λ[s, 3] +
            λ[s, 4] - λ[s, 5]
        ) <= dual1[1, s]
    )

    @constraint(
        IOP,
        [s = 1:NoS],
        2 * Q[s, 3] * hat_FH[s] + Q[s, 4] - (u[s] - T3 - 170 + b[s]) * λ[s, 2] -
        (u[s] - 393) * λ[s, 3] - (u[s] - 313) * λ[s, 4] + (u[s] - 323) * λ[s, 5] <=
        dual1[2, s]
    )
    @constraint(
        IOP,
        [s = 1:NoS],
        -(
            2 * Q[s, 3] * hat_FH[s] + Q[s, 4] - (u[s] - T3 - 170 + b[s]) * λ[s, 2] -
            (u[s] - 393) * λ[s, 3] - (u[s] - 313) * λ[s, 4] + (u[s] - 323) * λ[s, 5]
        ) <= dual1[2, s]
    )

    @constraint(IOP, [s = 1:NoS], hat_Qc[s] / 2 + 553 - T3 >= 0)
    @constraint(
        IOP,
        [s = 1:NoS],
        -10 - hat_Qc[s] +
        (u[s] - T3 - 170) * hat_FH[s] +
        a[s] * hat_Qc[s] +
        b[s] * hat_FH[s] >= dual3[s]
    )
    @constraint(IOP, [s = 1:NoS], 2 * T3 - 786 - hat_Qc[s] + (u[s] - 393) * hat_FH[s] >= 0)
    @constraint(IOP, [s = 1:NoS], 2 * T3 - 1026 - hat_Qc[s] + (u[s] - 313) * hat_FH[s] >= 0)
    @constraint(IOP, [s = 1:NoS], 2 * T3 - 1026 - hat_Qc[s] + (u[s] - 323) * hat_FH[s] <= 0)

    @constraint(IOP, [s = 1:NoS], a[s] == sum(p[1, j] * u[s]^(j - 1) for j = 1:degree))
    @constraint(IOP, [s = 1:NoS], b[s] == sum(p[2, j] * u[s]^(j - 1) for j = 1:degree))

    @constraint(
        IOP,
        [s = 1:NoS, i = 1:4],
        Q[s, i] == sum(q[i, j] * u[s]^(j - 1) for j = 1:degree)
    )

    @constraint(IOP, [s = 1:NoS], w1[s] == a[s] * λ[s, 2])
    @constraint(IOP, [s = 1:NoS], w2[s] == b[s] * λ[s, 2])

    @constraint(IOP, [s = 1:NoS], (hat_Qc[s] / 2 + 553 - T3) * λ[s, 1] <= dual2[1, s])
    @constraint(IOP, [s = 1:NoS], -((hat_Qc[s] / 2 + 553 - T3) * λ[s, 1]) <= dual2[1, s])

    @constraint(
        IOP,
        [s = 1:NoS],
        (
            -λ[s, 2] * 10 - λ[s, 2] * hat_Qc[s] +
            (u[s] - T3 - 170) * λ[s, 2] * hat_FH[s] +
            w1[s] * hat_Qc[s] +
            w2[s] * hat_FH[s]
        ) <= dual2[2, s]
    )
    @constraint(
        IOP,
        [s = 1:NoS],
        -(
            -λ[s, 2] * 10 - λ[s, 2] * hat_Qc[s] +
            (u[s] - T3 - 170) * λ[s, 2] * hat_FH[s] +
            w1[s] * hat_Qc[s] +
            w2[s] * hat_FH[s]
        ) <= dual2[2, s]
    )

    @constraint(
        IOP,
        [s = 1:NoS],
        (2 * T3 - 786 - hat_Qc[s] + (u[s] - 393) * hat_FH[s]) * λ[s, 3] <= dual2[3, s]
    )
    @constraint(
        IOP,
        [s = 1:NoS],
        -((2 * T3 - 786 - hat_Qc[s] + (u[s] - 393) * hat_FH[s]) * λ[s, 3]) <= dual2[3, s]
    )

    @constraint(
        IOP,
        [s = 1:NoS],
        ((2 * T3 - 1026 - hat_Qc[s] + (u[s] - 313) * hat_FH[s]) * λ[s, 4]) <= dual2[4, s]
    )
    @constraint(
        IOP,
        [s = 1:NoS],
        -((2 * T3 - 1026 - hat_Qc[s] + (u[s] - 313) * hat_FH[s]) * λ[s, 4]) <= dual2[4, s]
    )

    @constraint(
        IOP,
        [s = 1:NoS],
        ((2 * T3 - 1026 - hat_Qc[s] + (u[s] - 323) * hat_FH[s]) * λ[s, 5]) <= dual2[5, s]
    )
    @constraint(
        IOP,
        [s = 1:NoS],
        -((2 * T3 - 1026 - hat_Qc[s] + (u[s] - 323) * hat_FH[s]) * λ[s, 5]) <= dual2[5, s]
    )
    @constraint(IOP, [i = 1:2, j = 1:degree], p[i, j] <= t[i, j])
    @constraint(IOP, [i = 1:2, j = 1:degree], -p[i, j] <= t[i, j])

    @constraint(IOP, [i = 1:4, j = 1:degree], q[i, j] <= tt[i, j])
    @constraint(IOP, [i = 1:4, j = 1:degree], -q[i, j] <= tt[i, j])

    @constraint(IOP, [s = 1:NoS], Q[s, 3] >= 1)
    @constraint(IOP, [s = 1:NoS], Q[s, 1] >= 1)

    @objective(
        IOP,
        Min,
        100 * sum((hat_Qc[s] - x[s][1])^2 for s = 1:NoS) +
        100 * sum((hat_FH[s] - x[s][2])^2 for s = 1:NoS) +
        (sum(dual1) + sum(dual2) - sum(dual3)) +
        1e-5 * (sum(t) + sum(tt))
    )
    optimize!(IOP)
    if termination_status(IOP) == (MOI.NUMERICAL_ERROR) ||
       (termination_status(IOP) == (MOI.LOCALLY_INFEASIBLE)) ||
       (termination_status(IOP) == MOI.ITERATION_LIMIT)
        println(termination_status(IOP))
        global flag = 1
        println("something is wrong!")
    end

    for s = 1:NoS
        vars.primal[s][1] = value(hat_Qc[s])
        vars.primal[s][2] = value(hat_FH[s])
    end
    vars.slacks .= [value.(dual1), value.(dual2), abs.(value.(dual3))]
    vars.obj_params = value.(q)
    vars.dual .= value.(λ)
    vars.constraint_params .= value.(p)
end
