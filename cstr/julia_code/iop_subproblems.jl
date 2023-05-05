function Step1(S, inlet_A, inlet_R, a, Qc, lambda, c1, c2)
    IOP = Model(optimizer_with_attributes(Gurobi.Optimizer))
    set_optimizer_attribute(IOP, "OutputFlag", 0)
    set_optimizer_attribute(IOP, "BarHomogeneous", 1)
    set_optimizer_attribute(IOP, "NumericFocus", 3)

    JuMP.@variables(IOP, begin
        Ahat[1:S] >= 0
        Rhat[1:S] >= 0
        That[1:S] >= 0
        Tihat[1:S] >= 0
        delta1[1:S, 1:4] >= 0
        delta2[1:S, 1:3] >= 0
    end)

    @constraint(
        IOP,
        [s = 1:S],
        lambda[s, 1] * (-1 / tau - a[s, 2]) +
        lambda[s, 2] * (a[s, 2]) +
        lambda[s, 3] * (-dH / (rho * Cp) * a[s, 2]) <= delta1[s, 1]
    )
    @constraint(
        IOP,
        [s = 1:S],
        lambda[s, 1] * (-1 / tau - a[s, 2]) +
        lambda[s, 2] * (a[s, 2]) +
        lambda[s, 3] * (-dH / (rho * Cp) * a[s, 2]) >= -delta1[s, 1]
    )

    @constraint(
        IOP,
        [s = 1:S],
        2 * Qc[1] * Rhat[s] +
        Qc[2] +
        lambda[s, 1] * (a[s, 4]) +
        lambda[s, 2] * (-1 / tau - a[s, 4]) +
        lambda[s, 3] * (-dH / (rho * Cp) * (-a[s, 4])) <= delta1[s, 2]
    )
    @constraint(
        IOP,
        [s = 1:S],
        2 * Qc[1] * Rhat[s] +
        Qc[2] +
        lambda[s, 1] * (a[s, 4]) +
        lambda[s, 2] * (-1 / tau - a[s, 4]) +
        lambda[s, 3] * (-dH / (rho * Cp) * (-a[s, 4])) >= -delta1[s, 2]
    )

    @constraint(
        IOP,
        [s = 1:S],
        lambda[s, 1] * (-a[s, 1] + a[s, 3]) +
        lambda[s, 2] * (a[s, 1] - a[s, 3]) +
        lambda[s, 3] * (-dH / (rho * Cp) * (a[s, 1] - a[s, 3]) - 1 / tau) <= delta1[s, 3]
    )
    @constraint(
        IOP,
        [s = 1:S],
        lambda[s, 1] * (-a[s, 1] + a[s, 3]) +
        lambda[s, 2] * (a[s, 1] - a[s, 3]) +
        lambda[s, 3] * (-dH / (rho * Cp) * (a[s, 1] - a[s, 3]) - 1 / tau) >= -delta1[s, 3]
    )

    @constraint(
        IOP,
        [s = 1:S],
        2 * Qc[3] * Tihat[s] + Qc[4] + lambda[s, 3] * (1 / tau) <= delta1[s, 4]
    )
    @constraint(
        IOP,
        [s = 1:S],
        2 * Qc[3] * Tihat[s] + Qc[4] + lambda[s, 3] * (1 / tau) >= -delta1[s, 4]
    )


    @constraint(
        IOP,
        [s = 1:S],
        1 / tau * (inlet_A[s] - Ahat[s]) - (a[s, 1] * That[s] + a[s, 2] * Ahat[s]) +
        (a[s, 3] * That[s] + a[s, 4] * Rhat[s]) <= delta2[s, 1]
    )
    @constraint(
        IOP,
        [s = 1:S],
        1 / tau * (inlet_A[s] - Ahat[s]) - (a[s, 1] * That[s] + a[s, 2] * Ahat[s]) +
        (a[s, 3] * That[s] + a[s, 4] * Rhat[s]) >= -delta2[s, 1]
    )

    @constraint(
        IOP,
        [s = 1:S],
        1 / tau * (inlet_R - Rhat[s]) + (a[s, 1] * That[s] + a[s, 2] * Ahat[s]) -
        (a[s, 3] * That[s] + a[s, 4] * Rhat[s]) <= delta2[s, 2]
    )
    @constraint(
        IOP,
        [s = 1:S],
        1 / tau * (inlet_R - Rhat[s]) + (a[s, 1] * That[s] + a[s, 2] * Ahat[s]) -
        (a[s, 3] * That[s] + a[s, 4] * Rhat[s]) >= -delta2[s, 2]
    )

    @constraint(
        IOP,
        [s = 1:S],
        -dH / (rho * Cp) *
        (a[s, 1] * That[s] + a[s, 2] * Ahat[s] - (a[s, 3] * That[s] + a[s, 4] * Rhat[s])) +
        1 / tau * (Tihat[s] - That[s]) <= delta2[s, 3]
    )
    @constraint(
        IOP,
        [s = 1:S],
        -dH / (rho * Cp) *
        (a[s, 1] * That[s] + a[s, 2] * Ahat[s] - (a[s, 3] * That[s] + a[s, 4] * Rhat[s])) +
        1 / tau * (Tihat[s] - That[s]) >= -delta2[s, 3]
    )
    # @constraint(IOP, [s = 1:S], Ahat[s] + Rhat[s] == inlet_A[s])
    # @constraint(IOP, Qc[3] >= 1)

    @objective(
        IOP,
        Min,
        sum(
            (Ti_train[s] - Tihat[s])^2 +
            (A0_train[s] - Ahat[s])^2 +
            (R0_train[s] - Rhat[s])^2 +
            (T0_train[s] - That[s])^2 for s = 1:S
        ) +
        sum(c1 .* delta1) +
        sum(c2 .* delta2)
    )
    optimize!(IOP)
    if termination_status(IOP) != MOI.OPTIMAL
        global flag = 1
        return Ahat, Rhat, That, Tihat
    end
    return value.(Ahat), value.(Rhat), value.(That), value.(Tihat)
end

function Step2(S, inlet_A, inlet_R, a, Ahat, Rhat, That, Tihat, c1, c2)
    IOP = Model(optimizer_with_attributes(Gurobi.Optimizer))
    set_optimizer_attribute(IOP, "OutputFlag", 0)
    set_optimizer_attribute(IOP, "BarHomogeneous", 0)
    set_optimizer_attribute(IOP, "NumericFocus", 3)

    JuMP.@variables(IOP, begin
        Qc[1:4]
        delta1[1:S, 1:4] >= 0
        lambda[1:S, 1:3]
    end)

    @constraint(
        IOP,
        [s = 1:S],
        lambda[s, 1] * (-1 / tau - a[s, 2]) +
        lambda[s, 2] * (a[s, 2]) +
        lambda[s, 3] * (-dH / (rho * Cp) * a[s, 2]) <= delta1[s, 1]
    )
    @constraint(
        IOP,
        [s = 1:S],
        lambda[s, 1] * (-1 / tau - a[s, 2]) +
        lambda[s, 2] * (a[s, 2]) +
        lambda[s, 3] * (-dH / (rho * Cp) * a[s, 2]) >= -delta1[s, 1]
    )

    @constraint(
        IOP,
        [s = 1:S],
        2 * Qc[1] * Rhat[s] +
        Qc[2] +
        lambda[s, 1] * (a[s, 4]) +
        lambda[s, 2] * (-1 / tau - a[s, 4]) +
        lambda[s, 3] * (-dH / (rho * Cp) * (-a[s, 4])) <= delta1[s, 2]
    )
    @constraint(
        IOP,
        [s = 1:S],
        2 * Qc[1] * Rhat[s] +
        Qc[2] +
        lambda[s, 1] * (a[s, 4]) +
        lambda[s, 2] * (-1 / tau - a[s, 4]) +
        lambda[s, 3] * (-dH / (rho * Cp) * (-a[s, 4])) >= -delta1[s, 2]
    )

    @constraint(
        IOP,
        [s = 1:S],
        lambda[s, 1] * (-a[s, 1] + a[s, 3]) +
        lambda[s, 2] * (a[s, 1] - a[s, 3]) +
        lambda[s, 3] * (-dH / (rho * Cp) * (a[s, 1] - a[s, 3]) - 1 / tau) <= delta1[s, 3]
    )
    @constraint(
        IOP,
        [s = 1:S],
        lambda[s, 1] * (-a[s, 1] + a[s, 3]) +
        lambda[s, 2] * (a[s, 1] - a[s, 3]) +
        lambda[s, 3] * (-dH / (rho * Cp) * (a[s, 1] - a[s, 3]) - 1 / tau) >= -delta1[s, 3]
    )

    @constraint(
        IOP,
        [s = 1:S],
        2 * Qc[3] * Tihat[s] + Qc[4] + lambda[s, 3] * (1 / tau) <= delta1[s, 4]
    )
    @constraint(
        IOP,
        [s = 1:S],
        2 * Qc[3] * Tihat[s] + Qc[4] + lambda[s, 3] * (1 / tau) >= -delta1[s, 4]
    )
    @constraint(IOP, Qc[3] >= 1)
    @constraint(IOP, Qc[1] >= 0)

    # @constraint(IOP, Qc[3] >= 1)

    # @constraint(IOP, Qc[3] >= 1)

    @objective(IOP, Min, sum(c1 .* delta1))
    optimize!(IOP)
    if termination_status(IOP) != MOI.OPTIMAL
        global flag = 1
        return Qc, lambda
    end
    return value.(Qc), value.(lambda)
end

function Step3(S, inlet_A, inlet_R, Ahat, Rhat, That, Tihat, lambda, Qc, c1, c2, type)
    IOP = Model(optimizer_with_attributes(Gurobi.Optimizer))
    set_optimizer_attribute(IOP, "OutputFlag", 0)
    set_optimizer_attribute(IOP, "BarHomogeneous", 1)
    set_optimizer_attribute(IOP, "NumericFocus", 3)

    JuMP.@variables(IOP, begin
        a[1:S, 1:4] >= 0
        delta1[1:S, 1:4] >= 0
        delta2[1:S, 1:3] >= 0
        p[1:4, 1:4]
        posp[1:4, 1:4] >= 0
    end)

    @constraint(
        IOP,
        [s = 1:S],
        lambda[s, 1] * (-1 / tau - a[s, 2]) +
        lambda[s, 2] * (a[s, 2]) +
        lambda[s, 3] * (-dH / (rho * Cp) * a[s, 2]) <= delta1[s, 1]
    )
    @constraint(
        IOP,
        [s = 1:S],
        lambda[s, 1] * (-1 / tau - a[s, 2]) +
        lambda[s, 2] * (a[s, 2]) +
        lambda[s, 3] * (-dH / (rho * Cp) * a[s, 2]) >= -delta1[s, 1]
    )

    @constraint(
        IOP,
        [s = 1:S],
        2 * Qc[1] * Rhat[s] +
        Qc[2] +
        lambda[s, 1] * (a[s, 4]) +
        lambda[s, 2] * (-1 / tau - a[s, 4]) +
        lambda[s, 3] * (-dH / (rho * Cp) * (-a[s, 4])) <= delta1[s, 2]
    )
    @constraint(
        IOP,
        [s = 1:S],
        2 * Qc[1] * Rhat[s] +
        Qc[2] +
        lambda[s, 1] * (a[s, 4]) +
        lambda[s, 2] * (-1 / tau - a[s, 4]) +
        lambda[s, 3] * (-dH / (rho * Cp) * (-a[s, 4])) >= -delta1[s, 2]
    )

    @constraint(
        IOP,
        [s = 1:S],
        lambda[s, 1] * (-a[s, 1] + a[s, 3]) +
        lambda[s, 2] * (a[s, 1] - a[s, 3]) +
        lambda[s, 3] * (-dH / (rho * Cp) * (a[s, 1] - a[s, 3]) - 1 / tau) <= delta1[s, 3]
    )
    @constraint(
        IOP,
        [s = 1:S],
        lambda[s, 1] * (-a[s, 1] + a[s, 3]) +
        lambda[s, 2] * (a[s, 1] - a[s, 3]) +
        lambda[s, 3] * (-dH / (rho * Cp) * (a[s, 1] - a[s, 3]) - 1 / tau) >= -delta1[s, 3]
    )

    @constraint(
        IOP,
        [s = 1:S],
        2 * Qc[3] * Tihat[s] + Qc[4] + lambda[s, 3] * (1 / tau) <= delta1[s, 4]
    )
    @constraint(
        IOP,
        [s = 1:S],
        2 * Qc[3] * Tihat[s] + Qc[4] + lambda[s, 3] * (1 / tau) >= -delta1[s, 4]
    )

    @constraint(
        IOP,
        [s = 1:S],
        1 / tau * (inlet_A[s] - Ahat[s]) - (a[s, 1] * That[s] + a[s, 2] * Ahat[s]) +
        (a[s, 3] * That[s] + a[s, 4] * Rhat[s]) <= delta2[s, 1]
    )
    @constraint(
        IOP,
        [s = 1:S],
        1 / tau * (inlet_A[s] - Ahat[s]) - (a[s, 1] * That[s] + a[s, 2] * Ahat[s]) +
        (a[s, 3] * That[s] + a[s, 4] * Rhat[s]) >= -delta2[s, 1]
    )

    @constraint(
        IOP,
        [s = 1:S],
        1 / tau * (inlet_R - Rhat[s]) + (a[s, 1] * That[s] + a[s, 2] * Ahat[s]) -
        (a[s, 3] * That[s] + a[s, 4] * Rhat[s]) <= delta2[s, 2]
    )
    @constraint(
        IOP,
        [s = 1:S],
        1 / tau * (inlet_R - Rhat[s]) + (a[s, 1] * That[s] + a[s, 2] * Ahat[s]) -
        (a[s, 3] * That[s] + a[s, 4] * Rhat[s]) >= -delta2[s, 2]
    )

    @constraint(
        IOP,
        [s = 1:S],
        -dH / (rho * Cp) *
        (a[s, 1] * That[s] + a[s, 2] * Ahat[s] - (a[s, 3] * That[s] + a[s, 4] * Rhat[s])) +
        1 / tau * (Tihat[s] - That[s]) <= delta2[s, 3]
    )
    @constraint(
        IOP,
        [s = 1:S],
        -dH / (rho * Cp) *
        (a[s, 1] * That[s] + a[s, 2] * Ahat[s] - (a[s, 3] * That[s] + a[s, 4] * Rhat[s])) +
        1 / tau * (Tihat[s] - That[s]) >= -delta2[s, 3]
    )

    @constraint(
        IOP,
        [s = 1:S],
        a[s, 1] ==
        p[1, 1] * inlet_A[s]^3 + p[1, 2] * inlet_A[s]^2 + p[1, 3] * inlet_A[s] + p[1, 4]
    )
    @constraint(
        IOP,
        [s = 1:S],
        a[s, 2] ==
        p[2, 1] * inlet_A[s]^3 + p[2, 2] * inlet_A[s]^2 + p[2, 3] * inlet_A[s] + p[2, 4]
    )
    @constraint(
        IOP,
        [s = 1:S],
        a[s, 3] ==
        p[3, 1] * inlet_A[s]^3 + p[3, 2] * inlet_A[s]^2 + p[3, 3] * inlet_A[s] + p[3, 4]
    )
    @constraint(
        IOP,
        [s = 1:S],
        a[s, 4] ==
        p[4, 1] * inlet_A[s]^3 + p[4, 2] * inlet_A[s]^2 + p[4, 3] * inlet_A[s] + p[4, 4]
    )

    if type == 1
        @constraint(IOP, [i = 1:4, j = 1:4], p[i, j] <= posp[i, j])
        @constraint(IOP, [i = 1:4, j = 1:4], -p[i, j] <= posp[i, j])
    elseif type == 3
        @constraint(IOP, [i = 1:4], p[i, 1] == 0)
    elseif type == 4
        @constraint(IOP, [i = 1:4], p[i, 1] == 0)
        @constraint(IOP, [i = 1:4], p[i, 2] == 0)
    elseif type == 5
        @constraint(IOP, [i = 1:4], p[i, 1] == 0)
        @constraint(IOP, [i = 1:4], p[i, 2] == 0)
        @constraint(IOP, [i = 1:4], p[i, 3] == 0)
    end


    @objective(
        IOP,
        Min,
        sum(
            (Ti_train[s] - Tihat[s])^2 +
            (A0_train[s] - Ahat[s])^2 +
            (R0_train[s] - Rhat[s])^2 +
            (T0_train[s] - That[s])^2 for s = 1:S
        ) +
        sum(c1 .* delta1) +
        sum(c2 .* delta2) +
        1e-5 * sum(posp)
    )
    optimize!(IOP)
    if termination_status(IOP) != MOI.OPTIMAL
        global flag = 1
        return A, P, zeros(Float64, S, 4), zeros(Float64, S, 3), 0.0
    end
    return value.(a), value.(p), value.(delta1), value.(delta2), objective_value(IOP)
end
