# McCulloch–Pitts Based Logic Gates + Half Adder + Full Adder

# --------------------------
# McCulloch–Pitts GATES
# --------------------------

# def mc_and(x1, x2):
#     # weights
#     w1, w2 = 1, 1
#     # threshold
#     theta = 2

#     # weighted sum
#     net = x1*w1 + x2*w2

#     # activation function (Heaviside step)
#     if net >= theta:
#         return 1
#     else:
#         return 0


# # ---- MAIN ----
# print("McCulloch–Pitts AND Gate")

# for x1 in [0, 1]:
#     for x2 in [0, 1]:
#         print(f"{x1} AND {x2} = {mc_and(x1, x2)}")
    

def mp_and(x1, x2):
    """AND Gate using MP Neuron"""
    return 1 if (x1 + x2) >= 2 else 0   # threshold = 2


def mp_or(x1, x2):
    """OR Gate using MP Neuron"""
    return 1 if (x1 + x2) >= 1 else 0   # threshold = 1


def mp_not(x):
    """NOT Gate using MP Neuron"""
    return 1 if x == 0 else 0


# --------------------------
# XOR USING MP GATES
# XOR = (A OR B) AND NOT(A AND B)
# --------------------------

def mp_xor(a, b):
    or_ab = mp_or(a, b)
    and_ab = mp_and(a, b)
    not_and = mp_not(and_ab)
    return mp_and(or_ab, not_and)


# --------------------------
# HALF ADDER
# sum = A XOR B
# carry = A AND B
# --------------------------

def half_adder(a, b):
    sum_ = mp_xor(a, b)
    carry = mp_and(a, b)
    return sum_, carry


# --------------------------
# FULL ADDER
# sum = XOR(XOR(A,B), Cin)
# carry = (A AND B) OR (B AND Cin) OR (A AND Cin)
# --------------------------

def full_adder(a, b, cin):
    s1 = mp_xor(a, b)
    sum_ = mp_xor(s1, cin)

    c1 = mp_and(a, b)
    c2 = mp_and(b, cin)
    c3 = mp_and(a, cin)

    carry = mp_or(mp_or(c1, c2), c3)

    return sum_, carry


# --------------------------
# MAIN TESTING SECTION
# --------------------------

if __name__ == "__main__":
    print("=== HALF ADDER TABLE ===")
    for a in [0, 1]:
        for b in [0, 1]:
            s, c = half_adder(a, b)
            print(f"A={a}, B={b} -> Sum={s}, Carry={c}")

    print("\n=== FULL ADDER TABLE ===")
    for a in [0, 1]:
        for b in [0, 1]:
            for cin in [0, 1]:
                s, c = full_adder(a, b, cin)
                print(f"A={a}, B={b}, Cin={cin} -> Sum={s}, Carry={c}")
