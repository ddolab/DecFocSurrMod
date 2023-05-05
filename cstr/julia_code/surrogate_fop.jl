function SurrogateRTO(inlet_A, inlet_R, P, Qc)

    SRTO = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    # set_optimizer_attribute(SRTO, "OutputFlag", 0)
    a = zeros(Float64, 4)
    for i = 1:4
        a[i] = P[i, 1] * inlet_A^3 + P[i, 2] * inlet_A^2 + P[i, 3] * inlet_A + P[i, 4]
    end

    JuMP.@variables(SRTO, begin
        A >= 0
        R >= 0
        T >= 400
        Ti >= 400
    end)

    @constraint(
        SRTO,
        0 == 1 / tau * (inlet_A - A) - (a[1] * T + a[2] * A) + (a[3] * T + a[4] * R)
    )
    @constraint(
        SRTO,
        0 == 1 / tau * (inlet_R - R) + (a[1] * T + a[2] * A) - (a[3] * T + a[4] * R)
    )
    @constraint(
        SRTO,
        0 ==
        -dH / (rho * Cp) * ((a[1] * T + a[2] * A) - (a[3] * T + a[4] * R)) +
        1 / tau * (Ti - T)
    )

    @objective(SRTO, Min, Qc[1] * R^2 + Qc[2] * R + Qc[3] * Ti^2 + Qc[4] * Ti)

    optimize!(SRTO)
    return value.(Ti), value.(A), value.(R), value.(T)
end
