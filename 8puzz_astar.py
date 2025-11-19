import heapq
from copy import deepcopy

goal_pos = {
    1:(0,0), 2:(0,1), 3:(0,2),
    4:(1,0), 5:(1,1), 6:(1,2),
    7:(2,0), 8:(2,1)
}

def manhattan(b):
    s = 0
    for i in range(3):
        for j in range(3):
            if b[i][j] != 0:
                x, y = goal_pos[b[i][j]]
                s += abs(x - i) + abs(y - j)
    return s

def find_zero(b):
    for i in range(3):
        for j in range(3):
            if b[i][j] == 0:
                return i, j

def board_to_string(b):
    return ''.join(str(b[i][j]) for i in range(3) for j in range(3))

def print_board(b):
    for row in b:
        print(*row)

def astar_manhattan(start):
    pq = []
    h = manhattan(start)
    heapq.heappush(pq, (h, 0, start))
    visited = set()

    while pq:
        f, depth, curr = heapq.heappop(pq)
        key = board_to_string(curr)
        if key in visited:
            continue
        visited.add(key)

        print(f"Depth: {depth} | Heuristic: {f - depth} | Cost = {f}")
        print_board(curr)
        print()

        if manhattan(curr) == 0:
            print("Goal Reached at Depth", depth)
            print_board(curr)
            return

        x, y = find_zero(curr)
        moves = [(-1,0),(0,-1),(1,0),(0,1)]

        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                newb = deepcopy(curr)
                newb[x][y], newb[nx][ny] = newb[nx][ny], newb[x][y]
                h2 = manhattan(newb)
                if board_to_string(newb) not in visited:
                    heapq.heappush(pq, (depth + 1 + h2, depth+1, newb))


# -------- MAIN --------
print("Enter initial 8-puzzle configuration:")
board = [list(map(int, input().split())) for _ in range(3)]

astar_manhattan(board)
