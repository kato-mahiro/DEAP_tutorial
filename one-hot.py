import random

from deap import base
from deap import creator
from deap import tools


" 適応度の定義: base.Fitnessクラスを継承し、weights=(1.0,)というメンバ変数を追加した適応度を表す FitnessMax というクラスをcreatorモジュール内に作成している"
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

" 個体の定義: base.Fitnessクラスを継承して、 fitness=creator.FitnessMaxというメンバ変数を追加したIndividualクラスを定義している"
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

"toolbox.register関数で、引数のデフォルト値を設定する"
toolbox.register("attr_bool", random.randint, 0, 1)
" 個体を作成する関数を定義する"
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 100)
" 世代を作成する関数を定義する"
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    return sum(individual),

def myMutFlibBit(individual,indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = type(individual[i])(not individual[i])
    return individual,

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", myMutFlibBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    random.seed(64)
    
    pop = toolbox.population(n=300)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40
    
    print("Start of evolution")

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    print("  Evaluated %i individuals" % len(pop))
    
    for g in range(NGEN):
        print("-- Generation %i --" % g)
        
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        print("  Evaluated %i individuals" % len(invalid_ind))
        
        pop[:] = offspring
        
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
    
    print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

if __name__ == "__main__":
    main()
