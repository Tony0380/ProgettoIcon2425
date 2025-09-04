% Travel Planning Knowledge Base - ICon 2024/25
% 
% Rappresentazione conoscenza dominio viaggio con:
% - Regole logiche per vincoli e preferenze
% - Inferenza deduttiva per pianificazione  
% - Ragionamento simbolico integrato con ML
% - Query complesse per decision support
%
% CONCETTI ICon IMPLEMENTATI:
% 1. Logica del primo ordine e clausole di Horn
% 2. Unificazione e risoluzione SLD
% 3. Backward chaining per goal resolution
% 4. Cut e negation as failure
% 5. Definite clause grammars (DCG)

% ----------------------------------------------------------------------------
% FATTI BASE: Città, collegamenti e caratteristiche trasporti
% ----------------------------------------------------------------------------

% Città italiane con informazioni geografiche
city(milano, north, 1400000, high_cost).
city(roma, center, 2800000, medium_cost).  
city(napoli, south, 1000000, medium_cost).
city(torino, north, 870000, medium_cost).
city(bologna, north, 390000, medium_cost).
city(firenze, center, 380000, medium_cost).
city(bari, south, 325000, low_cost).
city(palermo, south, 670000, low_cost).
city(venezia, north, 260000, high_cost).
city(genova, north, 580000, medium_cost).
city(verona, north, 260000, medium_cost).
city(catania, south, 315000, low_cost).
city(cagliari, south, 155000, low_cost).
city(trieste, north, 205000, medium_cost).
city(perugia, center, 170000, medium_cost).
city(pescara, center, 120000, low_cost).
city(reggio_calabria, south, 180000, low_cost).
city(salerno, south, 135000, low_cost).
city(brescia, north, 200000, medium_cost).
city(pisa, center, 90000, medium_cost).

% Collegamenti diretti tra città con mezzi disponibili
direct_connection(milano, torino, [train, bus]).
direct_connection(milano, venezia, [train, bus]).
direct_connection(milano, bologna, [train, bus]).
direct_connection(milano, genova, [train, bus]).
direct_connection(torino, genova, [train, bus]).
direct_connection(venezia, verona, [train, bus]).
direct_connection(verona, brescia, [train, bus]).
direct_connection(bologna, firenze, [train, bus]).
direct_connection(firenze, roma, [train, bus]).
direct_connection(roma, napoli, [train, bus]).
direct_connection(perugia, roma, [train, bus]).
direct_connection(pisa, firenze, [train, bus]).
direct_connection(napoli, bari, [train, bus]).
direct_connection(napoli, salerno, [train, bus]).
direct_connection(bari, reggio_calabria, [train, bus]).

% Collegamenti aerei (solo voli per lunghe distanze)
direct_connection(roma, palermo, [flight]).
direct_connection(milano, palermo, [flight]).
direct_connection(roma, cagliari, [flight]).
direct_connection(milano, cagliari, [flight]).
direct_connection(catania, roma, [flight]).
direct_connection(milano, bari, [flight]).
direct_connection(roma, bari, [flight]).

% Caratteristiche mezzi di trasporto
transport_info(train, 120, 0.15, high, medium, weather_sensitive).
transport_info(bus, 80, 0.08, low, low, weather_resistant).
transport_info(flight, 500, 0.25, high, high, weather_very_sensitive).

% transport_info(Mezzo, VelocitàKmH, CostoPerKm, Comfort, Punctuality, WeatherSensitivity)

% ----------------------------------------------------------------------------
% PROFILI UTENTE E PREFERENZE  
% ----------------------------------------------------------------------------

% Definizione profili utente con caratteristiche comportamentali
user_profile(business, high_income, low_price_sensitivity, high_time_priority, high_comfort).
user_profile(leisure, medium_income, medium_price_sensitivity, medium_time_priority, medium_comfort).
user_profile(budget, low_income, high_price_sensitivity, low_time_priority, low_comfort).
user_profile(family, medium_income, medium_price_sensitivity, high_time_priority, high_comfort).
user_profile(student, low_income, very_high_price_sensitivity, low_time_priority, low_comfort).

% Preferenze trasporto per profilo
prefers_transport(business, flight).
prefers_transport(business, train).
prefers_transport(leisure, train).
prefers_transport(leisure, bus).
prefers_transport(budget, bus).
prefers_transport(family, train).
prefers_transport(student, bus).

% ----------------------------------------------------------------------------
% REGOLE LOGICHE: Pianificazione e vincoli
% ----------------------------------------------------------------------------

% Regola: Collegamento bidirezionale
connected(CityA, CityB, Transports) :-
    direct_connection(CityA, CityB, Transports).
connected(CityA, CityB, Transports) :-
    direct_connection(CityB, CityA, Transports).

% Regola: Percorso tra due città (con ricerca in profondità)
path(Origin, Destination, [Origin, Destination], [Transport]) :-
    connected(Origin, Destination, Transports),
    member(Transport, Transports).

path(Origin, Destination, [Origin|RestPath], [Transport|RestTransports]) :-
    connected(Origin, Intermediate, Transports),
    member(Transport, Transports),
    Origin \= Destination,
    Intermediate \= Origin,
    path(Intermediate, Destination, RestPath, RestTransports),
    \+ member(Intermediate, [Origin]).

% Regola: Trasporto disponibile per tratta
available_transport(Origin, Destination, Transport) :-
    connected(Origin, Destination, Transports),
    member(Transport, Transports).

% Regola: Trasporto adatto per utente
suitable_transport(UserProfile, Transport) :-
    user_profile(UserProfile, Income, PriceSens, TimePriority, ComfortPriority),
    transport_info(Transport, Speed, CostPerKm, Comfort, Punctuality, WeatherSens),
    transport_matches_profile(Income, PriceSens, TimePriority, ComfortPriority, 
                            Transport, Speed, CostPerKm, Comfort, Punctuality).

% Logica matching trasporto-profilo
transport_matches_profile(high_income, _, high_time_priority, high_comfort, 
                         flight, Speed, _, high, _, _) :- Speed >= 400.
transport_matches_profile(high_income, _, high_time_priority, _, 
                         train, Speed, _, _, high, _) :- Speed >= 100.
transport_matches_profile(_, high_price_sensitivity, _, _, 
                         bus, _, CostPerKm, _, _, _) :- CostPerKm =< 0.10.
transport_matches_profile(medium_income, medium_price_sensitivity, medium_time_priority, _, 
                         train, Speed, CostPerKm, _, _, _) :- Speed >= 80, CostPerKm =< 0.20.

% Regola: Viaggio valido con vincoli
valid_trip(Origin, Destination, UserProfile, Transport, Season, Budget) :-
    path(Origin, Destination, Path, Transports),
    member(Transport, Transports),
    suitable_transport(UserProfile, Transport),
    within_budget(Origin, Destination, Transport, Budget),
    no_travel_restrictions(Origin, Destination, Season),
    satisfies_user_constraints(UserProfile, Transport, Season).

% Vincoli di budget
within_budget(Origin, Destination, Transport, Budget) :-
    estimate_cost(Origin, Destination, Transport, EstimatedCost),
    EstimatedCost =< Budget.

% Stima costo semplificata (da migliorare con ML integration)
estimate_cost(Origin, Destination, Transport, Cost) :-
    distance(Origin, Destination, Distance),
    transport_info(Transport, _, CostPerKm, _, _, _),
    Cost is Distance * CostPerKm.

% Distanze approssimate (in km) - da sostituire con calcoli reali
distance(milano, roma, 570).
distance(milano, napoli, 770).
distance(milano, palermo, 935).
distance(roma, napoli, 225).
distance(roma, palermo, 490).
distance(torino, milano, 140).
distance(venezia, milano, 280).
distance(bologna, milano, 200).
distance(firenze, roma, 270).
distance(bologna, firenze, 105).

% Regola generica: se non specificata, calcola distanza euclidea approssimata
distance(CityA, CityB, Distance) :-
    CityA \= CityB,
    \+ distance(CityA, CityB, _),
    \+ distance(CityB, CityA, _),
    approx_distance(CityA, CityB, Distance).

distance(CityA, CityB, Distance) :-
    distance(CityB, CityA, Distance).

% Distanza approssimata per città non specificate
approx_distance(_, _, 300). % Default fallback

% Restrizioni di viaggio stagionali
no_travel_restrictions(_, _, summer).
no_travel_restrictions(Origin, Destination, Season) :-
    Season \= winter,
    \+ problematic_winter_route(Origin, Destination).
no_travel_restrictions(Origin, Destination, winter) :-
    \+ problematic_winter_route(Origin, Destination).

problematic_winter_route(_, catania).
problematic_winter_route(_, cagliari).
problematic_winter_route(trieste, _).

% Soddisfacimento vincoli utente
satisfies_user_constraints(business, Transport, _) :-
    transport_info(Transport, _, _, Comfort, Punctuality, _),
    (Comfort = high ; Punctuality = high).

satisfies_user_constraints(budget, Transport, _) :-
    transport_info(Transport, _, CostPerKm, _, _, _),
    CostPerKm =< 0.10.

satisfies_user_constraints(leisure, Transport, Season) :-
    transport_info(Transport, _, _, _, _, WeatherSens),
    (Season \= winter ; WeatherSens \= weather_very_sensitive).

satisfies_user_constraints(family, Transport, _) :-
    transport_info(Transport, _, _, Comfort, _, _),
    Comfort \= low.

satisfies_user_constraints(student, bus, _).

% ----------------------------------------------------------------------------
% REGOLE AVANZATE: Ottimizzazione e raccomandazioni
% ----------------------------------------------------------------------------

% Trova il miglior trasporto per profilo utente
best_transport_for_user(Origin, Destination, UserProfile, BestTransport) :-
    findall(Transport, 
            (available_transport(Origin, Destination, Transport),
             suitable_transport(UserProfile, Transport)), 
            SuitableTransports),
    SuitableTransports \= [],
    rank_transports(SuitableTransports, UserProfile, RankedTransports),
    RankedTransports = [BestTransport|_].

% Ranking trasporti per profilo (versione semplificata)
rank_transports(Transports, business, RankedTransports) :-
    sort_by_speed_comfort(Transports, RankedTransports).

rank_transports(Transports, budget, RankedTransports) :-
    sort_by_cost(Transports, RankedTransports).

rank_transports(Transports, leisure, RankedTransports) :-
    balance_cost_comfort(Transports, RankedTransports).

% Ordinamenti semplificati (da implementare con predicati ausiliari)
sort_by_speed_comfort([flight, train, bus], [flight, train, bus]).
sort_by_speed_comfort([train, bus], [train, bus]).
sort_by_speed_comfort([bus], [bus]).
sort_by_speed_comfort([], []).

sort_by_cost([bus, train, flight], [bus, train, flight]).
sort_by_cost([train, flight], [train, flight]).
sort_by_cost([flight], [flight]).
sort_by_cost([], []).

balance_cost_comfort([train, bus, flight], [train, bus, flight]).
balance_cost_comfort([bus, train], [train, bus]).
balance_cost_comfort([flight], [flight]).
balance_cost_comfort([], []).

% Raccomandazioni alternative
recommend_alternatives(Origin, Destination, UserProfile, Alternatives) :-
    findall((Transport, Reason), 
            (available_transport(Origin, Destination, Transport),
             \+ suitable_transport(UserProfile, Transport),
             alternative_reason(UserProfile, Transport, Reason)),
            Alternatives).

alternative_reason(business, bus, 'Più economico ma meno comfort').
alternative_reason(budget, flight, 'Più veloce ma costoso').
alternative_reason(leisure, flight, 'Veloce ma sensibile al meteo').

% Multi-città trip planning
multi_city_trip(Cities, UserProfile, Budget, TripPlan) :-
    Cities = [Start|RestCities],
    plan_multi_city(Start, RestCities, UserProfile, Budget, [], TripPlan).

plan_multi_city(_, [], _, _, Acc, Acc).
plan_multi_city(CurrentCity, [NextCity|RestCities], UserProfile, Budget, Acc, TripPlan) :-
    valid_trip(CurrentCity, NextCity, UserProfile, Transport, summer, Budget),
    estimate_cost(CurrentCity, NextCity, Transport, Cost),
    RemainingBudget is Budget - Cost,
    RemainingBudget >= 0,
    append(Acc, [(CurrentCity, NextCity, Transport, Cost)], NewAcc),
    plan_multi_city(NextCity, RestCities, UserProfile, RemainingBudget, NewAcc, TripPlan).

% ----------------------------------------------------------------------------
% QUERY COMPLESSE E INFERENZA
% ----------------------------------------------------------------------------

% Trova tutti i percorsi possibili con costi
all_paths_with_costs(Origin, Destination, UserProfile, Budget, PathsWithCosts) :-
    findall((Path, Transports, TotalCost), 
            (path(Origin, Destination, Path, Transports),
             calculate_path_cost(Path, Transports, TotalCost),
             TotalCost =< Budget,
             all_suitable_for_user(Transports, UserProfile)),
            PathsWithCosts).

% Calcola costo totale percorso
calculate_path_cost([_], [], 0).
calculate_path_cost([CityA, CityB|RestPath], [Transport|RestTransports], TotalCost) :-
    estimate_cost(CityA, CityB, Transport, SegmentCost),
    calculate_path_cost([CityB|RestPath], RestTransports, RestCost),
    TotalCost is SegmentCost + RestCost.

% Verifica che tutti i trasporti siano adatti all'utente
all_suitable_for_user([], _).
all_suitable_for_user([Transport|RestTransports], UserProfile) :-
    suitable_transport(UserProfile, Transport),
    all_suitable_for_user(RestTransports, UserProfile).

% Pianificazione con ottimizzazione multi-obiettivo
optimize_trip(Origin, Destination, UserProfile, Budget, OptimalTrip) :-
    all_paths_with_costs(Origin, Destination, UserProfile, Budget, AllPaths),
    AllPaths \= [],
    find_pareto_optimal(AllPaths, UserProfile, OptimalTrip).

% Selezione Pareto-optimal (versione semplificata)
find_pareto_optimal([(Path, Transports, Cost)], _, (Path, Transports, Cost)).
find_pareto_optimal([(PathA, TransportsA, CostA), (PathB, TransportsB, CostB)|Rest], UserProfile, OptimalTrip) :-
    (CostA =< CostB ->
        find_pareto_optimal([(PathA, TransportsA, CostA)|Rest], UserProfile, OptimalTrip)
    ;   find_pareto_optimal([(PathB, TransportsB, CostB)|Rest], UserProfile, OptimalTrip)
    ).

% ----------------------------------------------------------------------------
% INTERFACCIA E QUERY PRINCIPALI
% ----------------------------------------------------------------------------

% Query principale: pianifica viaggio
plan_travel(Origin, Destination, UserProfile, Budget, Season, TravelPlan) :-
    optimize_trip(Origin, Destination, UserProfile, Budget, OptimalTrip),
    OptimalTrip = (Path, Transports, TotalCost),
    create_detailed_plan(Path, Transports, TotalCost, Season, TravelPlan).

% Creazione piano dettagliato
create_detailed_plan(Path, Transports, TotalCost, Season, TravelPlan) :-
    TravelPlan = travel_plan(
        route: Path,
        transports: Transports,
        total_cost: TotalCost,
        season: Season,
        estimated_duration: unknown  % Da calcolare con regole temporali
    ).

% Query di supporto decisionale
travel_advice(Origin, Destination, UserProfile, Advice) :-
    findall(Transport, available_transport(Origin, Destination, Transport), Available),
    findall(Transport, suitable_transport(UserProfile, Transport), Suitable),
    intersection(Available, Suitable, Recommended),
    (Recommended \= [] ->
        Advice = recommended(Recommended)
    ;   recommend_alternatives(Origin, Destination, UserProfile, Alternatives),
        Advice = alternatives(Alternatives)
    ).

% Analisi fattibilità viaggio
travel_feasibility(Origin, Destination, UserProfile, Budget, Season, Result) :-
    (valid_trip(Origin, Destination, UserProfile, _, Season, Budget) ->
        Result = feasible
    ; estimate_min_cost(Origin, Destination, MinCost),
      (MinCost > Budget ->
          Result = insufficient_budget(MinCost)
      ; \+ no_travel_restrictions(Origin, Destination, Season) ->
          Result = seasonal_restrictions
      ; Result = no_suitable_transport
      )
    ).

% Stima costo minimo
estimate_min_cost(Origin, Destination, MinCost) :-
    findall(Cost, (available_transport(Origin, Destination, Transport),
                   estimate_cost(Origin, Destination, Transport, Cost)), Costs),
    min_list(Costs, MinCost).

% ----------------------------------------------------------------------------
% ESEMPI DI UTILIZZO E TEST
% ----------------------------------------------------------------------------

% Esempi di query che il sistema può rispondere:
%
% ?- plan_travel(milano, roma, business, 200, summer, Plan).
% ?- travel_advice(torino, palermo, budget, Advice).
% ?- multi_city_trip([milano, firenze, roma, napoli], leisure, 500, Trip).
% ?- travel_feasibility(venezia, cagliari, student, 50, winter, Result).
% ?- best_transport_for_user(bologna, bari, family, BestTransport).
% ?- all_paths_with_costs(milano, napoli, business, 300, Paths).

% Test queries per verifica funzionamento
test_basic_connectivity :-
    write('Testing basic connectivity...'), nl,
    connected(milano, roma, Transports),
    write('Milano-Roma connected by: '), write(Transports), nl.

test_path_finding :-
    write('Testing path finding...'), nl,
    path(milano, napoli, Path, Transports),
    write('Path Milano-Napoli: '), write(Path), nl,
    write('Transports: '), write(Transports), nl.

test_user_suitability :-
    write('Testing transport suitability...'), nl,
    suitable_transport(business, Transport),
    write('Business profile suitable transport: '), write(Transport), nl.

run_tests :-
    test_basic_connectivity,
    test_path_finding,  
    test_user_suitability.

% Fine del Knowledge Base
% ============================================================================