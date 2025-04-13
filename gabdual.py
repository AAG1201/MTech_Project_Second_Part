import numpy as np
from math import gcd, lcm

# def gabdual(g, a, M):
#     """
#     Compute the dual window for a Gabor frame.

#     Args:
#         g (numpy.ndarray): Input window (length L).
#         a (int): Analysis hop size.
#         M (int): Number of frequency channels.

#     Returns:
#         numpy.ndarray: Dual window.
#     """
#     # Length of the input window
#     Ls = len(g)

#     # Calculate the smallest common multiple of a and M
#     Lsmallest = lcm(a, M)
#     L = int(np.ceil(Ls / Lsmallest) * Lsmallest)

#     # Add zeros to make length of g equal to L if needed
#     Lfir = len(g)
#     if L > Lfir:
#         g = np.pad(g, (0, L - Lfir))

#     # Compute the diagonal
#     glong2 = np.abs(g)**2
#     N = L // a

#     # Initialize diagonal array
#     d = np.zeros(L, dtype=glong2.dtype)

#     # Compute a single period of the a-periodic diagonal
#     d[:a] = np.sum(glong2.reshape(N, a), axis=0) * M
#     d = np.tile(d[:a], N)

#     # Compute the dual window
#     gd = g / d[:M]

#     # Post-process the result to ensure it matches the input type
#     if np.isrealobj(g):
#         gd = np.real(gd)

#     return gd



def gabdual(g, a, M):
    """
    Compute the dual window for a Gabor frame.

    Args:
        g (numpy.ndarray): Input window (length L).
        a (int): Analysis hop size.
        M (int): Number of frequency channels.

    Returns:
        numpy.ndarray: Dual window (same length as original g).
    """
    import numpy as np
    from math import lcm

    Lfir = len(g)  # Save original length

    # Compute L as the smallest multiple of lcm(a, M) ≥ len(g)
    Lsmallest = lcm(a, M)
    L = int(np.ceil(Lfir / Lsmallest) * Lsmallest)

    # Zero-pad g if needed
    if L > Lfir:
        g = np.pad(g, (0, L - Lfir))

    # Compute diagonal of frame operator
    glong2 = np.abs(g) ** 2
    N = L // a
    d = np.zeros(L, dtype=glong2.dtype)
    d[:a] = np.sum(glong2.reshape(N, a), axis=0) * M
    d = np.tile(d[:a], N)

    # Compute the dual window over full length
    gd = g / d

    # Match original window length
    gd = gd[:Lfir]

    if np.isrealobj(g):
        gd = np.real(gd)

    return gd

