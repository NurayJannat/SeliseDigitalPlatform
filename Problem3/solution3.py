def iterate_matrix(map):
    count = 0
    for i in range(len(map)):
        # print("i", i)
        for j in range(len(map[i])):
            # print("j", j)
            # print("item: ", map[i][j])
            if map[i][j] == '0':
                pass
            
            elif map[i][j]=='1':
                # print("starting bro")
                find_island(map, i, j)
                count +=1

            elif map[i][j] == 'x':
                pass

    return count


def find_island(map, i, j):
    # print("find island", i, j)
    if map[i][j]=='1':
        map[i][j]='x'
        if i<len(map)-1:
            find_island(map, i+1, j)
        if i>0:
            find_island(map, i-1, j)
        if j<len(map[0])-1:
            find_island(map, i, j+1)
        if j>0:
            find_island(map, i, j-1)
    


if __name__ == "__main__":
    # map = [
    #     [1, 1, 1, 1, 0],
    #     [1, 1, 0, 1, 0],
    #     [1, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0]
    # ]

    map = []

    inp_list = input().split()

    while(inp_list!=[]):
        map.append(inp_list)
        inp_list = input().split()
        


print(iterate_matrix(map))