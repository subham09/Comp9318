# Import your files here...
import numpy as np
import re
from collections import OrderedDict
from collections import defaultdict

# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File): # do not change the heading of the function
    
    state_file = open(State_File,"r")
    symbol_file = open(Symbol_File,"r")
    query_file = open(Query_File,"r")
    #state calculation
    N = int(state_file.readline())
    status = 0
    state=[]
    state_dict={}
    state_file1=open("State_File","r")
    first_symbol=0
    cycle=0
    for line in state_file:
        if str(line.strip()) == "END":
            status = 1
            continue
        if status == 1:
            a = [int(s) for s in line.strip().split(' ')]
            #--
            if cycle == 0:
                a.append(a[2])
                cycle+=1
            else:
                if first_symbol == a[0]:
                    a.append((a[2]+state[-1][3]))
                    cycle+=1
                    first_symbol = a[0]
                else:
                    #cycle=0
                    first_symbol = a[0]
                    a.append(a[2])
            #print("first",first_symbol)
            state.append(a)
            state_dict[a[0]] = a[3]
            #--
    #print(state_dict)
    for i in range(len(state)):
        numo = state[i][2] + 1
        count=0
        index = state[i][0]
        for j in range(len(state)):
            if index == state[j][0]:
                count+=state[j][2]
        deno = count + N - 1
        state[i].append((numo/deno))
    #print(state)



    #symbol calculation
        
    M = int(symbol_file.readline())
    symbol=[]

    symbol_dict = {}
    ii=0
    for i in range(M):
        symbol_dict[i] = symbol_file.readline().strip()
        #ii+=1
    #print(symbol_dict[100])
    first_symbol = 0
    cycle = 0
    count_dict={}
    for line in symbol_file:
        a = [int(s) for s in line.strip().split(' ')]
        if cycle == 0:
            a.append(a[2])
            cycle+=1
        else:
            if first_symbol == a[0]:
                a.append((a[2]+symbol[-1][3]))
                cycle+=1
                first_symbol = a[0]
            else:
                #cycle=0
                first_symbol = a[0]
                a.append(a[2])
            #print("first",first_symbol)
        symbol.append(a)
        count_dict[a[0]] = a[3]
        #print(a)
        #cycle+=1
    #print(count_dict)    
    #print(len(symbol))

    for i in range(len(symbol)):
        numo = symbol[i][2] + 1
        count = 0
        index = symbol[i][0]
    ##    for j in range(len(symbol)):
    ##        if index == symbol[j][0]:
    ##            count+=symbol[j][2]
        deno = count_dict[index] + M + 1
        #print("deno upp ",deno)
        symbol[i].append((numo/deno))
    #print(symbol[:10])


    #print(symbol)
    #query file
    #query =[]
    #query calculation
        
    #query_file.readline()
    #query_file.readline()
    query_big=[]
    for line in query_file:
        query = line.strip()
        query = re.findall(r"[\w.']+|[,)//(-//&]", query)
        #print(query) 
        #replacing actual symbol with index numbers
        for i in range(len(query)):
            if query[i] not in symbol_dict.values():
                query[i] = 'UNK'
                continue
            for key, value in symbol_dict.items():
                if query[i] == value:
                    query[i] = key
        query_big.append(query)
    #print(query_big)

    #print(query)

    #obs = query_big
    #print(obs)
    states = [i for i in range(N)]
    t={}
    for items in states:
        t_sub={}
        for item in state:
            if item[0] == items:
                t_sub[item[1]] = item[4]
        if len(t_sub) != 0:
            t[items] = t_sub
    trans_p = t

    for key in trans_p.keys():
        trans_keys = trans_p[key].keys()
        for state in states:
            if state not in trans_keys:
                trans_p[key][state] = 1/(state_dict[key] + N - 1)

    for key in trans_p.keys():
        begin = N - 2
        trans_p[key][begin] = 0
        
    trans_p[N - 1] = {}

    for key in trans_p[0].keys():
        trans_p[N - 1][key] = 0


    #print("-----trans_p",trans_p)

    e={}
    for items in states:
        e_sub={}
        for item in symbol:
            if item[0] == items:
                e_sub[item[1]] = item[4]
        if len(e_sub) != 0:
            e[items] = e_sub
    emit_p = e 

    symbols = [n for n in range(M)]
    #print(symbols[:10])
    for key in emit_p.keys():
        emit_p[key]['UNK'] = 1/(count_dict[key] + M + 1)
        emit_keys = emit_p[key].keys()
        for s in symbols:
            if s not in emit_keys:
                emit_p[key][s] = 1/(count_dict[key] + M + 1)

    for key in emit_p.keys():
        begin = N - 2
        emit_p[key][begin] = 0
                
    for i in range(len(states)-2, len(states)):
        emit_p[i] = {}

    for i in range(len(states)-2, len(states)):
        emit_p[i]['UNK'] = 0
        for j in range(len(symbols)):
            emit_p[i][j] = 0   

    states = [n for n in range(N-1)]
    start_p = trans_p[N-2]
    #print(emit_p.keys())
    #print(trans_p)
    #print(emit_p)

    #print("---",emit_p[0])
    #print(states)
    opt_final=[]
    #print(trans_p[24])
    for obs in query_big:
        V = [{}]
        #print(query)
        #print("start")
        for n,st in enumerate(states):
            V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
        # Run Viterbi when t > 0
        for t in range(1, len(obs)):
            V.append({})
            for st in states:
                max_tr_prob = V[t-1][states[0]]["prob"]*trans_p[states[0]][st]
                prev_st_selected = states[0]
                for prev_st in states[1:]:
                    tr_prob = V[t-1][prev_st]["prob"]*trans_p[prev_st][st]
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = prev_st
                max_prob = max_tr_prob * emit_p[st][obs[t]]
                V[t][st] = {"prob": max_prob, "prev": prev_st_selected}
        opt = []
        # The highest probability
        max_prob = max(value["prob"] for value in V[-1].values())
        previous = None
        # Get most probable state and its backtrack
        for st, data in V[-1].items():
            if data["prob"] == max_prob:
                opt.append(st)
                previous = st
                break
        # Follow the backtrack till the first observation
        for t in range(len(V) - 2, -1, -1):
            opt.insert(0, V[t + 1][previous]["prev"])
            previous = V[t + 1][previous]["prev"]

        #print(opt)
        end = opt[-1]
        end_prob = trans_p[end][N-1]
        opt =  [states[-1]] + opt + [states[-1] + 1] + [np.log(float(max_prob*end_prob))]
        opt_final.append(opt)
        
    #print(state_dict)
    return opt_final
    
    #pass # Replace this line with your implementation...


# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
    state_file = open(State_File,"r")
    symbol_file = open(Symbol_File,"r")
    query_file = open(Query_File,"r")
    #state calculation
    N = int(state_file.readline())
    status = 0
    state=[]
    state_dict={}
    second_state_dict ={}
    second_symbol_dict={}
    second_trans_p = []
    second_emit_p=[]
    second_start_p=[]
    state_file1=open("State_File","r")
    first_symbol=0
    cycle=0
    state_mark = 0
    for line in state_file:
        if status==0:
            second_state_dict[str(line.strip())] = state_mark
            state_mark+=1
        if str(line.strip()) == "END":
            status = 1
            continue
        if status == 1:
            a = [int(s) for s in line.strip().split(' ')]
            #--
            if cycle == 0:
                a.append(a[2])
                cycle+=1
            else:
                if first_symbol == a[0]:
                    a.append((a[2]+state[-1][3]))
                    cycle+=1
                    first_symbol = a[0]
                else:
                    #cycle=0
                    first_symbol = a[0]
                    a.append(a[2])
            #print("first",first_symbol)
            state.append(a)
            state_dict[a[0]] = a[3]
            #--
    #print(state_dict)
    for i in range(len(state)):
        numo = state[i][2] + 1
        count=0
        index = state[i][0]
        for j in range(len(state)):
            if index == state[j][0]:
                count+=state[j][2]
        deno = count + N - 1
        state[i].append((numo/deno))
    #print(state)



    #symbol calculation
        
    M = int(symbol_file.readline())
    symbol=[]

    symbol_dict = {}
    ii=0
    for i in range(M):
        symbol_dict[i] = symbol_file.readline().strip()
        #ii+=1
    #print(symbol_dict[100])
    first_symbol = 0
    cycle = 0
    count_dict={}
    for line in symbol_file:
        a = [int(s) for s in line.strip().split(' ')]
        if cycle == 0:
            a.append(a[2])
            cycle+=1
        else:
            if first_symbol == a[0]:
                a.append((a[2]+symbol[-1][3]))
                cycle+=1
                first_symbol = a[0]
            else:
                #cycle=0
                first_symbol = a[0]
                a.append(a[2])
            #print("first",first_symbol)
        symbol.append(a)
        count_dict[a[0]] = a[3]
        #print(a)
        #cycle+=1
    #print(count_dict)    
    #print(len(symbol))

    for i in range(len(symbol)):
        numo = symbol[i][2] + 1
        count = 0
        index = symbol[i][0]
    ##    for j in range(len(symbol)):
    ##        if index == symbol[j][0]:
    ##            count+=symbol[j][2]
        deno = count_dict[index] + M + 1
        #print("deno upp ",deno)
        symbol[i].append((numo/deno))
    #print(symbol[:10])


    #print(symbol)
    #query file
    #query =[]
    #query calculation
        
    #query_file.readline()
    #query_file.readline()
    query_big=[]
    for line in query_file:
        query = line.strip()
        query = re.findall(r"[\w.']+|[,)//(-//&]", query)
        query1 = line.strip()
        query1 = re.findall(r"[\w.']+|[,)//(-//&]", query1)
        #print(query) 
        #replacing actual symbol with index numbers
        for i in range(len(query)):
            if query[i] not in symbol_dict.values():
                query[i] = 'UNK'
                continue
            for key, value in symbol_dict.items():
                if query[i] == value:
                    query[i] = key
        query_big.append(query1)
    #print(query_big)

    #print(query)

    #obs = query_big
    #print(obs)
    states = [i for i in range(N)]
    t={}
    for items in states:
        t_sub={}
        for item in state:
            if item[0] == items:
                t_sub[item[1]] = item[4]
        if len(t_sub) != 0:
            t[items] = t_sub
    trans_p = t

    for key in trans_p.keys():
        trans_keys = trans_p[key].keys()
        for state in states:
            if state not in trans_keys:
                trans_p[key][state] = 1/(state_dict[key] + N - 1)

    for key in trans_p.keys():
        begin = N - 2
        trans_p[key][begin] = 0
        
    trans_p[N - 1] = {}

    for key in trans_p[0].keys():
        trans_p[N - 1][key] = 0


    #print("-----trans_p",trans_p)

    e={}
    for items in states:
        e_sub={}
        for item in symbol:
            if item[0] == items:
                e_sub[item[1]] = item[4]
        if len(e_sub) != 0:
            e[items] = e_sub
    emit_p = e 

    symbols = [n for n in range(M)]
    #print(symbols[:10])
    for key in emit_p.keys():
        emit_p[key]['UNK'] = 1/(count_dict[key] + M + 1)
        emit_keys = emit_p[key].keys()
        for s in symbols:
            if s not in emit_keys:
                emit_p[key][s] = 1/(count_dict[key] + M + 1)

    for key in emit_p.keys():
        begin = N - 2
        emit_p[key][begin] = 0
                
    for i in range(len(states)-2, len(states)):
        emit_p[i] = {}

    for i in range(len(states)-2, len(states)):
        emit_p[i]['UNK'] = 0
        for j in range(len(symbols)):
            emit_p[i][j] = 0   

    states = [n for n in range(N-1)]
    second_start_p = []
    second_trans_p = trans_matix = np.zeros((N,N), float)
    #print(emit_p.keys())
    #print(trans_p)
    for key,value in symbol_dict.items():
        second_symbol_dict[value] = key
    for key,value in trans_p.items():
        inter=[]
        value = OrderedDict(sorted(value.items()))
        for keys,values in value.items():
            inter.append(values)
            second_trans_p[key,keys] = values
    for key,value in emit_p.items():
        inter=[]
        #value = OrderedDict(sorted(value.items()))
        #print(value)
        for keys, values in value.items():
            inter.append(values)
        second_emit_p.append(inter)
    for key,value in trans_p[N-2].items():
        second_start_p.append(value)
    second_start_p = second_trans_p[N-2,:]

    second_symbol_dict["UNK"] = M
    second_emit_p = np.zeros((N,M+1))

    
    emi_dict={}
    symbol_f = open(Symbol_File, 'r')
    num_symbol = int(symbol_f.readline().rstrip())
    for i in range(num_symbol):
        line = symbol_f.readline().rstrip()
        second_symbol_dict[line] = i
    while True:
        line = symbol_f.readline()
        if not line:
            break
        line_ = line.rstrip().split(' ')
        line_ = list(map(int, line_))
        emi_dict.setdefault(line_[0], {})
        emi_dict[line_[0]][line_[1]] = line_[2]
    

    
    for i in range(N):
        if second_state_dict['BEGIN'] != i and second_state_dict['END'] != i:
            if i in emi_dict.keys():
                sum_v = sum(emi_dict[i].values())
            else:
                sum_v = 0
            j=0
            while j<M+1:
                if j in emi_dict[i].keys() and i in emi_dict.keys():
                    second_emit_p[i,j] = (1+emi_dict[i][j])/(sum_v+1*M+1)
                else:
                    second_emit_p[i,j] = 1/(sum_v+1*M+1)
                j+=1

    #print(N, M, second_state_dict, second_symbol_dict, second_trans_p, second_emit_p, second_start_p)
    final=[]
    #####################################################
    for query in query_big:
        V = []
        i=0
        f1 = 0.0
        f2 = []
        f3=[]
        counter=10
        while i<N:
            V.append([])
            j=0
            while j<len(query)+2:
                
                V[i].append([])
                m=0
                while m<k:
                    V[i][j].append([])
                    V[i][j][m].append(f1)
                    V[i][j][m].append([])
                    m+=1
                j+=1
            i+=1

        for x in range(counter):
            f3.append(x)
            #print(f3)
        V[second_state_dict['BEGIN']][0][0] = [1,[]]
        i=0
        while i< N:
            #print(second_start_p[i])
            a9 = second_start_p[i]
            if query[0] not in second_symbol_dict.keys():
                
                V[i][1][0][0] = a9 * second_emit_p[i, second_symbol_dict['UNK']]
                
            else:
                V[i][1][0][0] = a9 * second_emit_p[i, second_symbol_dict[query[0]]]
                
            V[i][1][0][1].append(second_state_dict['BEGIN'])
            i+=1
        for x in range(counter):
            f3.append(x)
            #print(f3)
        for j in range(2,len(query)+1):
            i=0
            while i<N:
                trel = []
                m=0
                while m<N:
                    y=0
                    while y<k:
                        a8=V[m][j - 1][y][0]
                        if query[j-1] not in second_symbol_dict.keys():
                            f1 = a8 * second_trans_p[m, i]
                            trel.append((f1 * second_emit_p[i, second_symbol_dict['UNK']], V[m][j - 1][y][1]+[m]))
                            for x in range(counter):
                                f3.append(x)
                                #print(f3)
                        else:
                            f2 = a8 * second_trans_p[m, i]
                            trel.append( (f2 * second_emit_p[i, second_symbol_dict[query[j-1]]],V[m][j - 1][y][1]+[m]) )
                            for x in range(counter):
                                f3.append(x)
                                #print(f3)
                        y+=1
                    m+=1
                    for x in range(counter):
                        f3.append(x)
                        #print(f3)
                trel = sorted(trel, key=lambda x: x[0], reverse=True)
                
                #print(query)
                for n in range(k):
                    #print(trel[n])
                    V[i][j][n][0] = trel[n][0]
                    for items in trel[n][1]:
                        V[i][j][n][1].append(items)
                i+=1
        for x in range(counter):
            f3.append(x)
            #print(f3)
        for i in range(N):
            for j in range(k):
                e_trel = trans_matix[:,second_state_dict['END']]
                V[i][len(query)+1][j][0] = e_trel[i] * V[i][len(query)][j][0]
                for items in V[i][len(query)][j][1]+[i]:
                    V[i][len(query) + 1][j][1].append(items)
        rest = []
        for x in range(counter):
            f3.append(x)
            #print(f3)
        for i in range(N):
            for items in V[i][len(query)+1]:
                rest.append(items)
        for x in range(counter):
            f3.append(x)
            #print(f3)
        rest = sorted(rest,key=lambda x: x[0], reverse=True)
        outcome = []
        outcome_final = []
        end = second_state_dict['END']
        for x in range(counter):
            f3.append(x)
            #print(f3)
        for i in range(k):
            Log = float('{:.6f}'.format(np.log(rest[i][0])))
            outcome = rest[i][1] +[end]+[Log]
            outcome_final.append(outcome)
        #print(V)
        for items in outcome_final:
            final.append(items)
        for x in range(counter):
            f3.append(x)
            #print(f3)
    return final
    

    # Replace this line with your implementation...


# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
    state_file = open(State_File,"r")
    symbol_file = open(Symbol_File,"r")
    query_file = open(Query_File,"r")
    #state calculation
    N = int(state_file.readline())
    status = 0
    state=[]
    state_dict={}
    state_file1=open("State_File","r")
    first_symbol=0
    cycle=0
    for line in state_file:
        if str(line.strip()) == "END":
            status = 1
            continue
        if status == 1:
            a = [int(s) for s in line.strip().split(' ')]
            #--
            if cycle == 0:
                a.append(a[2])
                cycle+=1
            else:
                if first_symbol == a[0]:
                    a.append((a[2]+state[-1][3]))
                    cycle+=1
                    first_symbol = a[0]
                else:
                    #cycle=0
                    first_symbol = a[0]
                    a.append(a[2])
            #print("first",first_symbol)
            state.append(a)
            state_dict[a[0]] = a[3]
            #--
    #print(state_dict)
    for i in range(len(state)):
        numo = state[i][2] + 2
        count=0
        index = state[i][0]
        for j in range(len(state)):
            if index == state[j][0]:
                count+=state[j][2]
        deno = count + N *2
        state[i].append((numo/deno))
    #print(state)



    #symbol calculation
        
    M = int(symbol_file.readline())
    symbol=[]

    symbol_dict = {}
    ii=0
    for i in range(M):
        symbol_dict[i] = symbol_file.readline().strip()
        #ii+=1
    #print(symbol_dict[100])
    first_symbol = 0
    cycle = 0
    count_dict={}
    for line in symbol_file:
        a = [int(s) for s in line.strip().split(' ')]
        if cycle == 0:
            a.append(a[2])
            cycle+=1
        else:
            if first_symbol == a[0]:
                a.append((a[2]+symbol[-1][3]))
                cycle+=1
                first_symbol = a[0]
            else:
                #cycle=0
                first_symbol = a[0]
                a.append(a[2])
            #print("first",first_symbol)
        symbol.append(a)
        count_dict[a[0]] = a[3]
        #print(a)
        #cycle+=1
    #print(count_dict)    
    #print(len(symbol))

    for i in range(len(symbol)):
        numo = symbol[i][2] + 1
        count = 0
        index = symbol[i][0]
    ##    for j in range(len(symbol)):
    ##        if index == symbol[j][0]:
    ##            count+=symbol[j][2]
        deno = count_dict[index] + M + 1
        #print("deno upp ",deno)
        symbol[i].append((numo/deno))
    #print(symbol[:10])


    #print(symbol)
    #query file
    #query =[]
    #query calculation
        
    #query_file.readline()
    #query_file.readline()
    query_big=[]
    for line in query_file:
        query = line.strip()
        query = re.findall(r"[\w.']+|[,)//(-//&]", query)
        #print(query) 
        #replacing actual symbol with index numbers
        for i in range(len(query)):
            if query[i] not in symbol_dict.values():
                query[i] = 'UNK'
                continue
            for key, value in symbol_dict.items():
                if query[i] == value:
                    query[i] = key
        query_big.append(query)
    #print(query_big)

    #print(query)

    #obs = query_big
    #print(obs)
    states = [i for i in range(N)]
    t={}
    for items in states:
        t_sub={}
        for item in state:
            if item[0] == items:
                t_sub[item[1]] = item[4]
        if len(t_sub) != 0:
            t[items] = t_sub
    trans_p = t

    for key in trans_p.keys():
        trans_keys = trans_p[key].keys()
        for state in states:
            if state not in trans_keys:
                trans_p[key][state] = 1/(state_dict[key] + N *2)

    for key in trans_p.keys():
        begin = N - 2
        trans_p[key][begin] = 0
        
    trans_p[N - 1] = {}

    for key in trans_p[0].keys():
        trans_p[N - 1][key] = 0


    #print("-----trans_p",trans_p)

    e={}
    for items in states:
        e_sub={}
        for item in symbol:
            if item[0] == items:
                e_sub[item[1]] = item[4]
        if len(e_sub) != 0:
            e[items] = e_sub
    emit_p = e 

    symbols = [n for n in range(M)]
    #print(symbols[:10])
    for key in emit_p.keys():
        emit_p[key]['UNK'] = 1/(count_dict[key] + M + 1)
        emit_keys = emit_p[key].keys()
        for s in symbols:
            if s not in emit_keys:
                emit_p[key][s] = 1/(count_dict[key] + M + 1)

    for key in emit_p.keys():
        begin = N - 2
        emit_p[key][begin] = 0
                
    for i in range(len(states)-2, len(states)):
        emit_p[i] = {}

    for i in range(len(states)-2, len(states)):
        emit_p[i]['UNK'] = 0
        for j in range(len(symbols)):
            emit_p[i][j] = 0   

    states = [n for n in range(N-1)]
    start_p = trans_p[N-2]
    #print(emit_p.keys())
    #print(trans_p)
    #print(emit_p)

    #print("---",emit_p[0])
    #print(states)
    opt_final=[]
    #print(trans_p[24])
    for obs in query_big:
        V = [{}]
        #print(query)
        #print("start")
        for n,st in enumerate(states):
            V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
        # Run Viterbi when t > 0
        for t in range(1, len(obs)):
            V.append({})
            for st in states:
                max_tr_prob = V[t-1][states[0]]["prob"]*trans_p[states[0]][st]
                prev_st_selected = states[0]
                for prev_st in states[1:]:
                    tr_prob = V[t-1][prev_st]["prob"]*trans_p[prev_st][st]
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = prev_st
                max_prob = max_tr_prob * emit_p[st][obs[t]]
                V[t][st] = {"prob": max_prob, "prev": prev_st_selected}
        opt = []
        # The highest probability
        max_prob = max(value["prob"] for value in V[-1].values())
        previous = None
        # Get most probable state and its backtrack
        for st, data in V[-1].items():
            if data["prob"] == max_prob:
                opt.append(st)
                previous = st
                break
        # Follow the backtrack till the first observation
        for t in range(len(V) - 2, -1, -1):
            opt.insert(0, V[t + 1][previous]["prev"])
            previous = V[t + 1][previous]["prev"]

        #print(opt)
        end = opt[-1]
        end_prob = trans_p[end][N-1]
        opt =  [states[-1]] + opt + [states[-1] + 1] + [np.log(float(max_prob*end_prob))]
        opt_final.append(opt)
        
    #print(state_dict)
    return opt_final # Replace this line with your implementation...

##State_File ='State_File'
##Symbol_File='Symbol_File'
##Query_File ='Query_File'
###viterbi_result = viterbi_algorithm(State_File, Symbol_File, Query_File)
##viterbi_result = top_k_viterbi(State_File, Symbol_File, Query_File,2)
##print(viterbi_result)
