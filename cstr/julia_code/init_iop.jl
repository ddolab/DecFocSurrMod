function initialize(S, inlet_A, inlet_R)
    init = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    # init = Model(optimizer_with_attributes(Gurobi.Optimizer))
    # set_optimizer_attribute(init, "OutputFlag", 1)
    # set_optimizer_attribute(init, "BarHomogeneous", 1)
    # set_optimizer_attribute(init, "NumericFocus", 3)


    Ahat = deepcopy(A0_train)
    Rhat = deepcopy(R0_train)
    That = deepcopy(T0_train)
    Tihat = deepcopy(Ti_train)

    JuMP.@variables(init, begin
        a[1:S, 1:4] >= 0
        p[1:4]
        q[1:4]
        r[1:4]
        t[1:4]
        delta2[1:S, 1:3] >= 0
    end)

    @constraint(
        init,
        [s = 1:S],
        1 / tau * (inlet_A[s] - Ahat[s]) - (a[s, 1] * That[s] + a[s, 2] * Ahat[s]) +
        (a[s, 3] * That[s] + a[s, 4] * Rhat[s]) <= delta2[s, 1]
    )
    @constraint(
        init,
        [s = 1:S],
        1 / tau * (inlet_A[s] - Ahat[s]) - (a[s, 1] * That[s] + a[s, 2] * Ahat[s]) +
        (a[s, 3] * That[s] + a[s, 4] * Rhat[s]) >= -delta2[s, 1]
    )

    @constraint(
        init,
        [s = 1:S],
        1 / tau * (inlet_R - Rhat[s]) + (a[s, 1] * That[s] + a[s, 2] * Ahat[s]) -
        (a[s, 3] * That[s] + a[s, 4] * Rhat[s]) <= delta2[s, 2]
    )
    @constraint(
        init,
        [s = 1:S],
        1 / tau * (inlet_R - Rhat[s]) + (a[s, 1] * That[s] + a[s, 2] * Ahat[s]) -
        (a[s, 3] * That[s] + a[s, 4] * Rhat[s]) >= -delta2[s, 2]
    )

    @constraint(
        init,
        [s = 1:S],
        -dH / (rho * Cp) *
        (a[s, 1] * That[s] + a[s, 2] * Ahat[s] - (a[s, 3] * That[s] + a[s, 4] * Rhat[s])) +
        1 / tau * (Tihat[s] - That[s]) <= delta2[s, 3]
    )
    @constraint(
        init,
        [s = 1:S],
        -dH / (rho * Cp) *
        (a[s, 1] * That[s] + a[s, 2] * Ahat[s] - (a[s, 3] * That[s] + a[s, 4] * Rhat[s])) +
        1 / tau * (Tihat[s] - That[s]) >= -delta2[s, 3]
    )

    @constraint(
        init,
        [s = 1:S],
        a[s, 1] == p[1] * inlet_A[s]^3 + p[2] * inlet_A[s]^2 + p[3] * inlet_A[s] + p[4]
    )
    @constraint(
        init,
        [s = 1:S],
        a[s, 2] == q[1] * inlet_A[s]^3 + q[2] * inlet_A[s]^2 + q[3] * inlet_A[s] + q[4]
    )
    @constraint(
        init,
        [s = 1:S],
        a[s, 3] == r[1] * inlet_A[s]^3 + r[2] * inlet_A[s]^2 + r[3] * inlet_A[s] + r[4]
    )
    @constraint(
        init,
        [s = 1:S],
        a[s, 4] == t[1] * inlet_A[s]^3 + t[2] * inlet_A[s]^2 + t[3] * inlet_A[s] + t[4]
    )

    @objective(
        init,
        Min,
        sum(delta2) +
        1e-5 * (
            sum(p[i]^2 for i = 1:4) +
            sum(q[i]^2 for i = 1:4) +
            sum(r[i]^2 for i = 1:4) +
            sum(t[i]^2 for i = 1:4)
        )
    )
    optimize!(init)
    return value.(a)
end
