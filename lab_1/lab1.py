import matplotlib.pyplot as plt
import numpy as np

T = 5 # Right border of the observed interval, [0, T]
delta = 0.01 # Step
N = round(T / delta) + 1 # Amount of observations
f_vals = [] # Input data observations that will be read from file
freq = 1 / T
hz = (N-1) * T / 100 # Frequency
found_index = None
dot_space = np.linspace(0, T, num=N, endpoint=True) # List of points for observations

def floor_zero(x):
    '''Converts negative values to zeros.'''

    return x if x >= 0 else 0

def main():
    '''Main function.'''

    global f_vals, dot_space, freq, found_index

    # Read input observations from file
    with open("f18.txt", "r") as f:
        raw = f.read()
        f_vals = [float(x) for x in raw.split(" ")]
        print(len(f_vals), N)
        f.close()

    # Prepare plots for input observations and fourier transformation on them
    plt.close()
    plt.figure("Fourier tranform constants finding lab")

    plt.subplot(211)
    plt.plot(dot_space, f_vals, 'g', label="Input data")
    plt.legend(loc="best")

    # Calculating Fourier transformation
    fft_vals = np.fft.fft(f_vals)

    plt.subplot(212)
    plt.plot(dot_space, list(map(floor_zero, fft_vals)), 'r--', label="Fourier transform of input data")
    plt.legend(loc="best")

    # Finiding first local maxima's index, excluding the left border one
    maxs = np.r_[True, fft_vals[1:] < fft_vals[:-1]] & np.r_[fft_vals[:-1] < fft_vals[1:], True]
    maxs_ind = [int(x * 100 - 1) for x in [a*b for a,b in zip(maxs, dot_space)]]

    for i in maxs_ind:
        if i >= 0:
            found_index = i + 1
            break

    ind_hz = int(found_index * freq)
    
    # y(t) = a1*t^3 + a2*t^2 + a3*t + a4*sin(2pi*ind_hz*t) + a5
    # Should be equal to observations in these points
    # So we solve the linear system with least squares criterion
    syst = []
    for x in dot_space:
        chronos = []
        chronos.append(x**3)
        chronos.append(x**2)
        chronos.append(x)
        chronos.append(np.sin(2*np.pi*ind_hz*x))
        chronos.append(1)
        syst.append(chronos)

    syst = np.array(syst, dtype=complex)
    
    raw_answer = list(map(np.real, np.linalg.lstsq(syst, f_vals)[0].tolist()))
    answer = list(map(lambda x: round(x, 4), raw_answer))
    
    # Printing out the answer and showing plots
    print("\nConstants:\na1: {}\na2: {}\na3: {}\na4: {}\na5: {}".format(*(answer)))

    plt.show()

if __name__ == "__main__":
    main()