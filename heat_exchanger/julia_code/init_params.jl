function init_objective_params(p)
    # This would also initialize the dual vars
    T3 = 388
    # Q = [0, 1e-2, 4, -3.4*4]
    IOP = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    JuMP.@variables(IOP, begin
        λ[1:NoS, 1:5] >= 0
        Q[1:NoS, 1:4]
        q[1:4, 1:degree]
        tt[1:4, 1:degree] >= 0
        slack1[1:2, 1:NoS] >= 0
        slack2[1:5, 1:NoS] >= 0
        a[1:NoS]
        b[1:NoS]
        w1[1:NoS]
        w2[1:NoS]
    end)
    @NLconstraint(
        IOP,
        [s = 1:NoS],
        2 * Q[s, 1] * x[s][1] + Q[s, 2] - 0.5 * λ[s, 1] - (a[s] - 1) * λ[s, 2] +
        λ[s, 3] +
        λ[s, 4] - λ[s, 5] <= slack1[1, s]
    )
    @NLconstraint(
        IOP,
        [s = 1:NoS],
        -(
            2 * Q[s, 1] * x[s][1] + Q[s, 2] - 0.5 * λ[s, 1] - (a[s] - 1) * λ[s, 2] +
            λ[s, 3] +
            λ[s, 4] - λ[s, 5]
        ) <= slack1[1, s]
    )

    @NLconstraint(
        IOP,
        [s = 1:NoS],
        2 * Q[s, 3] * x[s][2] + Q[s, 4] - (u[s] - T3 - 170 + b[s]) * λ[s, 2] -
        (u[s] - 393) * λ[s, 3] - (u[s] - 313) * λ[s, 4] + (u[s] - 323) * λ[s, 5] <=
        slack1[2, s]
    )
    @NLconstraint(
        IOP,
        [s = 1:NoS],
        -(
            2 * Q[s, 3] * x[s][2] + Q[s, 4] - (u[s] - T3 - 170 + b[s]) * λ[s, 2] -
            (u[s] - 393) * λ[s, 3] - (u[s] - 313) * λ[s, 4] + (u[s] - 323) * λ[s, 5]
        ) <= slack1[2, s]
    )

    @constraint(
        IOP,
        [s = 1:NoS, i = 1:4],
        Q[s, i] == sum(q[i, j] * u[s]^(j - 1) for j = 1:degree)
    )
    @constraint(IOP, [i = 1:4, j = 1:degree], q[i, j] <= tt[i, j])
    @constraint(IOP, [i = 1:4, j = 1:degree], -q[i, j] <= tt[i, j])
    @constraint(IOP, [s = 1:NoS], Q[s, 3] >= 1)
    @constraint(IOP, [s = 1:NoS], Q[s, 1] >= 1)

    @constraint(IOP, [s = 1:NoS], a[s] == sum(p[1, j] * u[s]^(j - 1) for j = 1:4))
    @constraint(IOP, [s = 1:NoS], b[s] == sum(p[2, j] * u[s]^(j - 1) for j = 1:4))

    @NLconstraint(IOP, [s = 1:NoS], w1[s] == a[s] * λ[s, 2])
    @NLconstraint(IOP, [s = 1:NoS], w2[s] == b[s] * λ[s, 2])

    @NLconstraint(IOP, [s = 1:NoS], (x[s][1] / 2 + 553 - T3) * λ[s, 1] <= slack2[1, s])
    @NLconstraint(IOP, [s = 1:NoS], -((x[s][1] / 2 + 553 - T3) * λ[s, 1]) <= slack2[1, s])

    @NLconstraint(
        IOP,
        [s = 1:NoS],
        (
            -λ[s, 2] * 10 - λ[s, 2] * x[s][1] +
            (u[s] - T3 - 170) * λ[s, 2] * x[s][2] +
            w1[s] * x[s][1] +
            w2[s] * x[s][2]
        ) <= slack2[2, s]
    )
    @NLconstraint(
        IOP,
        [s = 1:NoS],
        -(
            -λ[s, 2] * 10 - λ[s, 2] * x[s][1] +
            (u[s] - T3 - 170) * λ[s, 2] * x[s][2] +
            w1[s] * x[s][1] +
            w2[s] * x[s][2]
        ) <= slack2[2, s]
    )

    @NLconstraint(
        IOP,
        [s = 1:NoS],
        (2 * T3 - 786 - x[s][1] + (u[s] - 393) * x[s][2]) * λ[s, 3] <= slack2[3, s]
    )
    @NLconstraint(
        IOP,
        [s = 1:NoS],
        -((2 * T3 - 786 - x[s][1] + (u[s] - 393) * x[s][2]) * λ[s, 3]) <= slack2[3, s]
    )

    @NLconstraint(
        IOP,
        [s = 1:NoS],
        ((2 * T3 - 1026 - x[s][1] + (u[s] - 313) * x[s][2]) * λ[s, 4]) <= slack2[4, s]
    )
    @NLconstraint(
        IOP,
        [s = 1:NoS],
        -((2 * T3 - 1026 - x[s][1] + (u[s] - 313) * x[s][2]) * λ[s, 4]) <= slack2[4, s]
    )

    @NLconstraint(
        IOP,
        [s = 1:NoS],
        ((2 * T3 - 1026 - x[s][1] + (u[s] - 323) * x[s][2]) * λ[s, 5]) <= slack2[5, s]
    )
    @NLconstraint(
        IOP,
        [s = 1:NoS],
        -((2 * T3 - 1026 - x[s][1] + (u[s] - 323) * x[s][2]) * λ[s, 5]) <= slack2[5, s]
    )
    # @constraint(IOP, Q[2] >= 1)
    # @constraint(IOP, Q[1] >= 0)
    @constraint(IOP, sum(Q) == 100)

    @objective(IOP, Min, sum(slack1) + sum(slack2) + 1e-5 * sum(tt))
    optimize!(IOP)
    temp_λ = value.(λ)
    # λ = [temp_λ[s, :] for s in 1:NoS]
    q = value.(q)
    return (q), (temp_λ)
end

function init_constraint_params()
    T3 = 388
    Init = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    JuMP.@variables(Init, begin
        p[1:2, 1:degree]
        a[1:NoS]
        b[1:NoS]
        slack[1:NoS] <= 0
        t[1:2, 1:degree] >= 0
    end)

    @constraint(
        Init,
        [s = 1:NoS],
        -10 - x[s][1] + (u[s] - T3 - 170) * x[s][2] + a[s] * x[s][1] + b[s] * x[s][2] >=
        slack[s]
    )

    @constraint(Init, [s = 1:NoS], a[s] == sum(p[1, j] * u[s]^(j - 1) for j = 1:4))
    @constraint(Init, [s = 1:NoS], b[s] == sum(p[2, j] * u[s]^(j - 1) for j = 1:4))
    @constraint(Init, [i = 1:2, j = 1:degree], p[i, j] <= t[i, j])
    @constraint(Init, [i = 1:2, j = 1:degree], -p[i, j] <= t[i, j])

    @objective(Init, Max, sum(slack) - 1e-3 * sum(t))
    optimize!(Init)
    return value.(p)
end
