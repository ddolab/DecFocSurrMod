function RTO(inlet_A, inlet_R)

    MPC = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))

    JuMP.@variables(MPC, begin
        A >= 0
        R >= 0
        T >= 400
        Ti >= 400
        k1 >= 0
        km1 >= 0
    end)

    @NLconstraint(MPC, k1 == C1 * exp(-Q / (gc * T)))
    @NLconstraint(MPC, km1 == Cm1 * exp(-Qm1 / (gc * T)))

    @NLconstraint(MPC, 0 == 1 / tau * (inlet_A - A) - k1 * A + km1 * R)
    @NLconstraint(MPC, 0 == 1 / tau * (inlet_R - R) + k1 * A - km1 * R)
    @NLconstraint(MPC, 0 == -dH / (rho * Cp) * (k1 * A - km1 * R) + 1 / tau * (Ti - T))

    @objective(MPC, Min, -(2.009 * R - (1.657e-3 * (Ti - 410))^2))

    optimize!(MPC)
    return value.(Ti), value.(A), value.(R), value.(T)
end
