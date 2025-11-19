class Model:

    ip_descripters_speed = ["ss", "ms", "hs", "vhs"]
    ip_descripters_distance = ["sd", "md", "ld", "vld"]
    op_descripters = ["lp", "mp", "hp", "vhp"]

    start_values = {
        "sd": 0, "md": 100, "ld": 200, "vld": 500,
        "ss": 0, "ms": 5, "hs": 10, "vhs": 15,
        "lp": 0, "mp": 25, "hp": 50, "vhp": 75
    }

    mid_values = {
        "sd": 100, "md": 200, "ld": 500, "vld": 1000,
        "ss": 5, "ms": 10, "hs": 15, "vhs": 20,
        "lp": 25, "mp": 50, "hp": 75, "vhp": 100
    }

    end_values = {
        "sd": 200, "md": 500, "ld": 1000, "vld": 2000,
        "ss": 10, "ms": 15, "hs": 20, "vhs": 25,
        "lp": 50, "mp": 75, "hp": 100, "vhp": 125
    }

    # Mapping of (distance, speed) → output descriptor index
    cover = [
        [1, 2, 3, 3],
        [0, 1, 2, 2],
        [0, 0, 1, 1],
        [0, 0, 0, 1]
    ]

    # -----------------------------------------------
    # MEMBERSHIP FUNCTIONS
    # -----------------------------------------------
    def calculate_speed(self, speed, i):
        name = self.ip_descripters_speed[i]
        start = self.start_values[name]
        mid = self.mid_values[name]
        end = self.end_values[name]

        if speed < start or speed >= end:
            return 0
        elif speed < mid:
            return (speed - start) / (mid - start)
        else:
            return (end - speed) / (end - mid)

    def calculate_dist(self, distance, i):
        name = self.ip_descripters_distance[i]
        start = self.start_values[name]
        mid = self.mid_values[name]
        end = self.end_values[name]

        if distance < start or distance >= end:
            return 0
        elif distance < mid:
            return (distance - start) / (mid - start)
        else:
            return (end - distance) / (end - mid)

    # -----------------------------------------------
    # F U Z Z I F I C A T I O N
    # -----------------------------------------------
    def fuzzyfi(self, speed, distance):

        print("\n--- Fuzzification ---")
        speed_func = [self.calculate_speed(speed, i) for i in range(4)]
        dist_func = [self.calculate_dist(distance, i) for i in range(4)]

        # Build fuzzy matrix
        matrix = [[0]*4 for _ in range(4)]
        maxval = 0
        maxindex = (0, 0)

        for i in range(4):
            for j in range(4):
                matrix[i][j] = min(dist_func[i], speed_func[j])
                if matrix[i][j] > maxval:
                    maxval = matrix[i][j]
                    maxindex = (i, j)

        print("\nFuzzy Matrix:")
        for row in matrix:
            print(row)

        print("\nMost Activated Rule Strength =", maxval)
        print("Distance:", self.ip_descripters_distance[maxindex[0]])
        print("Speed:", self.ip_descripters_speed[maxindex[1]])

        output_label = self.op_descripters[self.cover[maxindex[0]][maxindex[1]]]
        print("Output Category =", output_label)

        # Defuzzification (average of linear interpolation)
        start = self.start_values[output_label]
        mid = self.mid_values[output_label]
        end = self.end_values[output_label]

        val1 = start + maxval * (mid - start)
        val2 = end - maxval * (mid - start)
        final = (val1 + val2) / 2

        print("Possible Value 1:", val1)
        print("Possible Value 2:", val2)
        print("\nFinal Defuzzified Output =", final)


# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
def main():
    fuzzy = Model()
    fuzzy.fuzzyfi(6.83, 100)


if __name__ == "__main__":
    main()
