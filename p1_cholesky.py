import numpy as np

def read_matrix(n: int) -> list[list[int]]:
    print("A:")
    A = []
    for i in range(n):
        row = input().split(' ')
        for j in range(n):
            row[j] = int(row[j])
        A.append(row)
    return A

def print_matrix(M: list[list[float]], name: str):
    print(name + ":")
    for row in M:
        print(" ".join(map(str, row)))

def cholesky(n: int, A: list[list[int]]) -> tuple[list[list[float]],bool]:
    '''
    Checks if matrix A is positive definite.
    If so, computes Cholesky factor G.
    Args:
        n (int): Dimension of matrix A.
        A (list[list[int]]): Matrix A.
    Returns:
        tuple: Contains:
            - G (list[list[float]]): Cholesky factor G.
            - positive (bool): Whether matrix A is positive definite.
    '''
    G = [[0 for _ in range(n)] for _ in range(n)]
    positive = True
    for i in range(n):
        for j in range(i+1):
            s = A[i][j]
            # Computes the sum g_ik.g_jk, k < j, then subtracts from a_ij
            for k in range(j):
                s -= G[i][k] * G[j][k]
            if j < i:
                G[i][j] = s / G[j][j]       # Divides by g_jj
            else:           # j == i
                if s > 0:   # Since g_ii is a positive real number
                    G[i][j] = np.sqrt(s)    # Computes the square root
                
                else:       # G does not exist, so A is not positive definite
                    positive = False
                    break
        if not positive:
            break
    return G, positive

def main():
    n = int(input("n: "))
    A = read_matrix(n)
    G, positive = cholesky(n, A)
    # Output
    if positive: print_matrix(G, "G")
    else: print("Error: A is not positive definite.")

if __name__ == "__main__":
    main()