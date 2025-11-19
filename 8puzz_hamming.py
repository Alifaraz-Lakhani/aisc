import heapq
from copy import deepcopy
import time

goal_pos = {
    1:(0,0), 2:(0,1), 3:(0,2),
    4:(1,0), 5:(1,1), 6:(1,2),
    7:(2,0), 8:(2,1)
}

def hamming(p):
    c = 0
    for i in range(3):
        for j in range(3):
            if p[i][j] != 0:
                if goal_pos[p[i][j]] != (i, j):
                    c += 1
    return c

def find_zero(p):
    for i in range(3):
        for j in range(3):
            if p[i][j] == 0:
                return i, j

def print_board(b):
    for row in b:
        print(*row)

def solve_missing_tiles(start):
    pq = []
    h = hamming(start)
    heapq.heappush(pq, (h, 0, start))
    start_t = time.time()

    while pq:
        f, depth, curr = heapq.heappop(pq)

        print("Depth:", depth)
        print("Hamming Distance:", f)
        print("Cost:", depth + f)
        print_board(curr)
        print()

        if f == 0:
            print("Goal Found at Depth:", depth)
            print_board(curr)
            print("Execution time:", int((time.time()-start_t)*1000), "ms")
            return

        x, y = find_zero(curr)
        moves = [(-1,0),(1,0),(0,-1),(0,1)]

        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                newp = deepcopy(curr)
                newp[x][y], newp[nx][ny] = newp[nx][ny], newp[x][y]
                h2 = hamming(newp)
                heapq.heappush(pq, (h2, depth+1, newp))


# -------- MAIN --------
print("Enter initial 8-puzzle state:")
p = [list(map(int, input().split())) for _ in range(3)]
solve_missing_tiles(p)
