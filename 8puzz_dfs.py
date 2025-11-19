from copy import deepcopy

goal_pos = {
    1:(0,0), 2:(0,1), 3:(0,2),
    4:(1,0), 5:(1,1), 6:(1,2),
    7:(2,0), 8:(2,1)
}

def is_goal(puzzle):
    for i in range(3):
        for j in range(3):
            if puzzle[i][j] != 0:
                if goal_pos[puzzle[i][j]] != (i, j):
                    return False
    return True

def find_zero(p):
    for i in range(3):
        for j in range(3):
            if p[i][j] == 0:
                return i, j
    return -1, -1

def dfs_8_puzzle(start):
    stack = [(start, 0)]
    max_depth = 31

    while stack:
        curr, depth = stack.pop()
        
        print("Depth:", depth)
        for row in curr:
            print(*row)
        print()

        if is_goal(curr):
            print("Goal Found at Depth:", depth)
            return

        if depth == max_depth:
            continue

        x, y = find_zero(curr)
        moves = [(-1,0),(1,0),(0,-1),(0,1)]  # up, down, left, right

        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                newp = deepcopy(curr)
                newp[x][y], newp[nx][ny] = newp[nx][ny], newp[x][y]
                stack.append((newp, depth + 1))


# -------- MAIN --------
puzzle = []
print("Enter initial 8-puzzle state (0 for blank):")
for _ in range(3):
    puzzle.append(list(map(int, input().split())))

dfs_8_puzzle(puzzle)
