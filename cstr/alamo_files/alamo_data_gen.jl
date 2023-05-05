using JuMP, DataFrames, Ipopt, CSV, BARON

tau = 60 # s
C1 = 5e3 # 1/s
Cm1 = 1e6 # 1/s
Q = 10000 # cal/mol
Qm1 = 15000 # cal/mol
gc = 1.987 # cal/mol.K
dH = -5000 # cal/mol
rho = 1.0 # kg/L
Cp = 1000 # cal/kg.K

function ALAMOSurrogate(inlet_A, instance)
    # MPC = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    MPC = Model(optimizer_with_attributes(BARON.Optimizer))
    @JuMP.variables(MPC, begin
    A >= 0
    R >= 0
    T >= 400
    Ti >= 400
    r
    end)

    if instance == 1
        @NLconstraint(MPC, r == - 9.6674517866367981611120 * A + 21.089076857998897196467 * R + 0.11816373751977610195851E-001 * A*T - 0.24853580746539160467590E-001 * R*T + 1987.6487778808855182433 * A/T - 4475.8512690759753240854 * R/T)
    elseif instance == 2
        @NLconstraint(MPC, r == - 9.5891604529477501017709 * A + 21.268255881651782601693 * R + 0.11725613224517124691393E-001 * A*T - 0.25049695859028268951629E-001 * R*T + 1970.7900548564966811682 * A/T - 4516.8179816182037029648 * R/T)
    elseif instance == 3
        @NLconstraint(MPC, r == - 9.8788243980671293087426 * A + 21.444164581125534141393 * R + 0.12046203034363475725677E-001 * A*T - 0.25253176552095073226223E-001 * R*T + 2035.9860917918117593217 * A/T - 4554.5965892711565174977 * R/T)
    elseif instance == 4
        @NLconstraint(MPC, r == - 9.6242219360080909495991 * A + 21.030464089648646108799 * R + 0.11775113365104101711966E-001 * A*T - 0.24813184481050902518984E-001 * R*T + 1976.5860703765099515294 * A/T - 4457.7365858963694336126 * R/T)
    elseif instance == 5
        @NLconstraint(MPC, r == - 10.039649513325752394621 * A + 21.310444817203702427832 * R + 0.12216294840567180901569E-001 * A*T - 0.25091415185098896872828E-001 * R*T + 2073.9395818801135646936 * A/T - 4527.2540301414564964944 * R/T)
    end
    @NLconstraint(MPC, 0 == 1/tau * (inlet_A - A) - r)
    @NLconstraint(MPC, 0 == 1/tau * (0 - R) + r)
    @NLconstraint(MPC, 0 == -dH/(rho*Cp) * (r) + 1/tau * (Ti - T))

    @objective(MPC, Min, -(2.009*R - (1.657e-3*(Ti - 410))^2))

    optimize!(MPC)

    return value.(Ti), JuMP.solve_time(MPC)
end

n_instance = 5
I_s = 5000
Ai_ss = 1
Ai = [Ai_ss + rand(rng, -8000:10000)/10000 for _ in 1:I_s]

temp = zeros(Float64, length(Ai))
time = zeros(Float64, n_instance, length(Ai))

for instance in 1:n_instance
    for j in 1:length(Ai)
        temp[instance, j], time[instance, j] = ALAMOSurrogate(Ai[j], instance)
    end
end

temp_df = DataFrame([Ai, temp], :auto)
CSV.write("alamo_output_temp.csv", temp_df)
time_df = DataFrame(time, :auto)
CSV.write("alamo_output_time.csv", time_df)
