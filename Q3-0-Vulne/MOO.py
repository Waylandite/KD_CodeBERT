import argparse
import csv
import time
import random
import logging
import numpy as np
import torch

from tqdm import tqdm
from transformers import RobertaTokenizer

from distill_utils import distill
from surrogate import predictor
from utils import mapFunction_convert
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)

# candidate object  containing the candidate values and the objective values
class Candidate(object):
    def __init__(self, candidates_vals):
        self.candidate_values = candidates_vals
        self.objective_values = []

    def get_candidate_values(self):
        return self.candidate_values

    def set_objective_values(self, objective_values):
        self.objective_values = objective_values

    def get_objective_values(self):
        return self.objective_values

    def set_candidate_values(self,candidates_vals):
        self.candidate_values = candidates_vals

    def set_candidate_values_at_index(self, indx, val):
        self.candidate_values[indx] = val


# generate random population of size 'size'
def generate_random_population(size, lb, ub):
    random_pop = []
    for i in range(size):
        while True:
            candidate_vals = []
            for index in range(len(lb)):
                if index==0:
                    candidate_vals.append(random.randint(lb[index], ub[index]))
                else :
                    candidate_vals.append(random.randint(max(lb[index],candidate_vals[index-1]+1), ub[index]))
                    # candidate_vals.append(random.randint(lb[index], ub[index]))
            # in this ,we can set the condition for the candidate values
            if True:
                break
        random_pop.append(Candidate(candidate_vals))

    return random_pop

# calculate the minimum distance between the candidate and the random population
def calculate_minimum_distance(candidate, random_pop):
    distance = 1e9
    for each_candidate in random_pop:
        vals = each_candidate.get_candidate_values()
        candidate_vals = candidate.get_candidate_values()
        dist = np.linalg.norm(np.array(vals) - np.array(candidate_vals))
        if dist < distance:
            distance = dist

    return distance

# generate adaptive random population
def generate_adaptive_random_population(size, lb, ub):
    random_pop = []

    initial_pop = generate_random_population(10, lb, ub)[0]

    random_pop.append(initial_pop)

    while len(random_pop) < size:
        Distance = 0
        selected_candidate = None
        rp = generate_random_population(size, lb, ub)
        # select max distance candidate  to add the diversity
        for each_candidate in rp:
            min_dis = calculate_minimum_distance(each_candidate, random_pop)
            if min_dis > Distance:
                Distance = min_dis
                selected_candidate = each_candidate

        random_pop.append(selected_candidate)

    return random_pop

# evaluate the population
def evaulate_population(population, surrogate_model_acc, surrogate_model_f1, surrogate_model_pre, surrogate_model_rec):
    for each_candidate in population:
        candidate_values = each_candidate.get_candidate_values()
        accuracy = surrogate_model_acc.predict([candidate_values])[0]
        f1 = surrogate_model_f1.predict([candidate_values])[0]
        precision = surrogate_model_pre.predict([candidate_values])[0]
        recall = surrogate_model_rec.predict([candidate_values])[0]
        fitnesses = [accuracy, f1, precision, recall]
        each_candidate.set_objective_values(fitnesses)

# eavaluate two candidate
def dominates(candidate1, candidate2):
    candidate1_objectives = candidate1.get_objective_values()
    candidate2_objectives = candidate2.get_objective_values()
    dominates = False
    if (candidate1_objectives[0] > candidate2_objectives[0]) and (candidate1_objectives[3] > candidate2_objectives[3]):
        dominates = True
    return dominates

# select the best candidates to archive
def update_archive(pop, archive):
    for each_candidate in pop:
        dominated = False
        for each_archive in archive:
            if dominates(each_archive, each_candidate):
                dominated = True
                break
        if not dominated:
            if len(archive) == 0:
                archive.append(each_candidate)
            else:
                to_remove = []
                # remove the dominated candidate
                for each_archive in archive:
                    if dominates(each_candidate,
                                 each_archive) or each_archive.get_candidate_values() == each_candidate.get_candidate_values():
                        to_remove.append(each_archive)

                for each_remove in to_remove:
                    archive.remove(each_remove)
                archive.append(each_candidate)

def is_non_decreasing(each_archive):
    calcalate=each_archive.get_candidate_values()
    calcalate=sorted(calcalate)
    each_archive.set_candidate_values(calcalate)
    return each_archive


# partially mapped crossover
def partially_mapped_crossover(parent1, parent2):
    parent1_values = parent1.get_candidate_values()
    parent2_values = parent2.get_candidate_values()
    size = len(parent1_values)
    child1 = [-1] * size
    child2 = [-1] * size

    crossover_point1, crossover_point2 = sorted(random.sample(range(size), 2))

    # for i in range(crossover_point1, crossover_point2 + 1):
    #     if parent2_values[i] not in child1:
    #         next_index = i
    #         while child1[next_index] != -1:
    #             next_index = parent2_values.index(parent1_values[next_index])
    #         child1[next_index] = parent2_values[i]

    #     if parent1_values[i] not in child2:
    #         next_index = i
    #         while child2[next_index] != -1:
    #             next_index = parent1_values.index(parent2_values[next_index])
    #         child2[next_index] = parent1_values[i]

    # for i in range(size):
    #     if child1[i] == -1:
    #         child1[i] = parent2_values[i]
    #     if child2[i] == -1:
    #         child2[i] = parent1_values[i]

    for i in range(size):
        if crossover_point1 <= i <= crossover_point2:
            child1[i] = parent2_values[i]
            child2[i] = parent1_values[i]
        else:
            child1[i] = parent1_values[i]
            child2[i] = parent2_values[i]
    

    # 由于交换的是对应位置，所以交换后的值一定是在范围内的，而进过排序后，各个位置也一定处在正确的范围内
    # return is_non_decreasing(Candidate(child1)), is_non_decreasing(Candidate(child2))
    return Candidate(child1), Candidate(child2)

# boundary random mutation
def boundary_random_mutation(candidate, lb, ub, thresh):
    candidate_values = candidate.get_candidate_values()
    for index in range(len(candidate_values)):
        if random.uniform(0, 1) < thresh :
            if candidate_values[index]>ub[index] or candidate_values[index]<lb[index] :
                candidate_values[index]=random.randint(lb[index], ub[index])
            # if index==0:
            #     if candidate_values[index+1]>lb[index]:
            #         candidate_values[index]=(random.randint(lb[index], min(ub[index],candidate_values[index+1]-1)))
            # elif index==len(candidate_values)-1:
            #     if candidate_values[index-1]<ub[index]:
            #         candidate_values[index]=(random.randint(max(lb[index],candidate_values[index-1]+1), ub[index]))
            # else :
            #     if candidate_values[index-1]< candidate_values[index+1]-1 :
            #         candidate_values[index] =random.randint(max(lb[index],candidate_values[index-1]+1), min(ub[index],candidate_values[index+1]-1))
    
    candidate.set_candidate_values(candidate_values)
    return candidate

# correct the population
def correct(pop, lb, ub):
    for indx in range(len(pop)):
        candidate = pop[indx]
        values = candidate.get_candidate_values()
        # set_values = []
        for value_index in range(len(values)):
            # pop[indx].set_candidate_values_at_index(value_index, int(pop[indx].get_candidate_values()[value_index]))
            while values[value_index] > ub[value_index] or values[value_index] < lb[value_index]:
                temp = generate_random_population(1, lb, ub)[0]
                pop[indx].set_candidate_values_at_index(value_index, int(temp.get_candidate_values()[value_index]))
                values = pop[indx].get_candidate_values()
            # while values[value_index] in set_values:
            #     temp = generate_random_population(1, lb, ub)[0]
            #     pop[indx].set_candidate_values_at_index(value_index, int(temp.get_candidate_values()[value_index]))
            #     values = pop[indx].get_candidate_values()
            # set_values.append(values[value_index]) 
        # pop[indx]=is_non_decreasing(pop[indx])
    return pop

# select the best candidate from the tournament candidates
def select_best(tournament_candidates):
    best = tournament_candidates[0]
    for i in range(len(tournament_candidates)):
        candidate1 = tournament_candidates[i]
        for j in range(i,len(tournament_candidates)):
            candidate2 = tournament_candidates[j]
            if (dominates(candidate1, candidate2)):
                best = candidate1
    return best

# select the best candidate from the random size population
def tournament_selection(pop, size):
    tournament_candidates = []
    for i in range(size):
        indx = random.randint(0, len(pop) - 1)
        random_candidate = pop[indx]
        tournament_candidates.append(random_candidate)

    best = select_best(tournament_candidates)
    return best

def is_correct(candidate):
    candidate_values = candidate.get_candidate_values()
    # 遍历value，判断是否递增
    for i in range(1, len(candidate_values)):
        if candidate_values[i] <= candidate_values[i - 1]:
            return False
    return True

# off spring generation contains the crossover and mutation
def generate_off_springs(pop, lb, ub):
    size = len(pop)
    population_to_return = []

    while len(population_to_return) < size:
        parent1 = tournament_selection(pop, 10)
        parent2 = tournament_selection(pop, 10)
        while parent1 == parent2:
            parent2 = tournament_selection(pop, 10)

        probability_crossover = random.uniform(0, 1)
        if probability_crossover <= 0.60:
            parent1, parent2 = partially_mapped_crossover(parent1, parent2)
        child1 = boundary_random_mutation(parent1, lb, ub, 0.1)
        child2 = boundary_random_mutation(parent2, lb, ub, 0.1)

        if(is_correct(child1)):
            population_to_return.append(child1)
        if(is_correct(child2)):
            population_to_return.append(child2)
        # population_to_return.append(child1)
        # population_to_return.append(child2)

    return population_to_return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        required=False,
                        help="The student model dir.")
    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        required=False,
                        help="The teacher model dir.")
    parser.add_argument("--tokenizer",
                        default=None,
                        type=str,
                        required=False,
                        help="The teacher model dir.")
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")

    parser.add_argument("--hid_epoches", type=int, default=42,
                        help="epoch num for hid training")
    parser.add_argument("--loss_function", type=str, default="mse",
                        help="loss function for attention")
    parser.add_argument("--pred_epoches", type=int, default=42,
                        help="epoch num for pred training")
    parser.add_argument("--hid_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for hid Adam.")
    parser.add_argument("--pred_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for pred Adam.")

    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--temperature',
                        type=float,
                        default=1.)
    parser.add_argument('--iteration',
                    type=int,
                    default=50)
    # prepare the device
    args = parser.parse_args()
    logger.info(args)
    args.device = torch.device("cuda")

    # prepare the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer)
    tokenizer.do_lower_case = True

    start_time = time.time()
    #cite the unidorm function and int() function
    # in this we set ub->max


    #  param3   mapfunction need to
    lb = [ 0, 0, 0, 0, 0, 0]
    ub = [ 12, 12, 12, 12, 12, 12]
    # logging.info("Lower Bound: {}".format(lb))
    # logging.info("Upper Bound: {}".format(ub))

    pop = generate_adaptive_random_population(20, lb, ub)

    for index in pop:
        logging.info(index.get_candidate_values())
    
    map_functionList = []
    for each_pop in pop:
        map_functionList.append(each_pop.get_candidate_values())

    # map_functionList=[[2,4,6,8,10,12]]
    # logging.info("Current map_functionList: {}".format(map_functionList))

    accs,f1s,pres,recs = distill(tokenizer,args,map_functionList, eval=False,surrogate=False)

    # accs = [round(random.uniform(0.5, 0.6), 2) for _ in range(20)]
    # f1s = [round(random.uniform(0.5, 0.6), 2) for _ in range(20)]
    # pres = [round(random.uniform(0.5, 0.6), 2) for _ in range(20)]
    # recs = [round(random.uniform(0.5, 0.6), 2) for _ in range(20)]


    with open("surrogate_train_data.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["first", "second", "third", "fourth", "fifth", "sixth", "Accuracy", "F1", "Precision", "Recall"])
        for d, acc,f1,pre,rec in zip(map_functionList, accs, f1s, pres, recs):
            writer.writerow(mapFunction_convert(d) + [acc] + [f1] + [pre] + [rec])

    surrogate_model_acc = predictor([map_functionList, accs])
    surrogate_model_f1 = predictor([map_functionList, f1s])
    surrogate_model_pre = predictor([map_functionList, pres])
    surrogate_model_rec = predictor([map_functionList, recs])

    # reload the surrogate model
    # import ast
    # with open("accs.jsonl") as f:
    #     data = f.readlines()
    #     surrogate_data_acc = []
    #     data1 = []
    #     data2 = []
    #     data3=[]
    #     data4=[]
    #     data5=[]
    #     for line in data:
    #         data1.append(ast.literal_eval(line.split("] ")[0]+"]"))
    #         data2.append(float(line.split("] ")[1]))
    #         data3.append(float(line.split("] ")[2]))
    #         data4.append(float(line.split("] ")[3]))
    #         data5.append(float(line.split("] ")[4]))
    #     print(data1)
    #     print(data2)
    #     print(data3)
    #     print(data4)
    #     print(data5)
    # surrogate_data_acc = [data1, data2]
    # surrogate_model = predictor(surrogate_data_acc)


#遗传算法流程
    evaulate_population(pop, surrogate_model_acc, surrogate_model_f1, surrogate_model_pre, surrogate_model_rec)
    archive = []
    update_archive(pop, archive)

    iteration = args.iteration
    for i in tqdm(range(iteration)):
        pop = generate_off_springs(pop, lb, ub)
        pop = correct(pop, lb, ub)
        evaulate_population(pop, surrogate_model_acc, surrogate_model_f1, surrogate_model_pre, surrogate_model_rec)
        update_archive(pop, archive)

    logging.info("Time taken: {}".format(time.time() - start_time))
    logging.info("Number of solutions in the archive: {}".format(len(archive)))
    logging.info("Saving the archive to the file")
    with open("pareto_set.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["first", "second", "third", "fourth", "fifth", "sixth", "Accuracy", "F1", "Precision", "Recall"])
        for each_archive in archive:
           # each_archive= is_non_decreasing(each_archive)
            writer.writerow(each_archive.get_candidate_values() + each_archive.get_objective_values())



