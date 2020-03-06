import numpy as np


class DFT:
    @staticmethod
    def slow_one_dimension(a):
        a = np.asarray(a, dtype=complex)
        N = a.shape[0]
        res = np.zeros(N, dtype=complex)

        for k in range(N):
            for n in range(N):
                res[k] += a[n] * np.exp(-2j * np.pi * k * n / N)

        return res

    @staticmethod
    def slow_one_dimension_inverse(a):
        a = np.asarray(a, dtype=complex)
        N = a.shape[0]
        res = np.zeros(N, dtype=complex)

        for n in range(N):
            for k in range(N):
                res[n] += a[k] * np.exp(2j * np.pi * k * n / N)

            res[n] /= N

        return res

    @staticmethod
    def fast_one_dimension(a):
        a = np.asarray(a, dtype=complex)
        N = a.shape[0]

        if N % 2 != 0:
            raise AssertionError("size of a must be a power of 2")
        elif N <= 4:
            return DFT.slow_one_dimension(a)
        else:
            even = DFT.fast_one_dimension(a[::2])
            odd = DFT.fast_one_dimension(a[1::2])
            res = np.zeros(N, dtype=complex)

            half_size = N//2
            for n in range(N):
                res[n] = even[n % half_size] + \
                    np.exp(-2j * np.pi * n / N) * odd[n % half_size]

            return res

    @staticmethod
    def fast_one_dimension_inverse(a):
        a = np.asarray(a, dtype=complex)
        N = a.shape[0]

        if N % 2 != 0:
            raise AssertionError("size of a must be a power of 2")
        elif N <= 4:
            return DFT.slow_one_dimension_inverse(a)
        else:
            even = DFT.fast_one_dimension_inverse(a[::2])
            odd = DFT.fast_one_dimension_inverse(a[1::2])
            res = np.zeros(N, dtype=complex)

            half_size = N//2
            for n in range(N):
                res[n] = half_size * even[n % half_size] + \
                    np.exp(2j * np.pi * n / N) * half_size * odd[n % half_size]
                res[n] /= N

            return res

    @staticmethod
    def slow_two_dimension(a):
        a = np.asarray(a, dtype=complex)
        N, M = a.shape
        res = np.zeros((N, M), dtype=complex)

        for k in range(N):
            for l in range(M):
                for m in range(M):
                    for n in range(N):
                        res[k, l] += a[n, m] * \
                            np.exp(-2j * np.pi * ((l * m / M) + (k * n / N)))

        return res

    @staticmethod
    def slow_two_dimension_inverse(a):
        a = np.asarray(a, dtype=complex)
        N, M = a.shape
        res = np.zeros((N, M), dtype=complex)

        for k in range(N):
            for l in range(M):
                for m in range(M):
                    for n in range(N):
                        res[k, l] += a[n, m] * \
                            np.exp(2j * np.pi * ((l * m / M) + (k * n / N)))
                res[k, l] /= M * N

        return res

    @staticmethod
    def fast_two_dimension(a):
        a = np.asarray(a, dtype=complex)
        N, M = a.shape

        if N % 2 != 0 or M % 2 != 0:
            raise AssertionError("size of a must be a power of 2")
        elif N <= 4 and M <= 4:
            return DFT.slow_two_dimension(a)
        else:
            # perform the operation row by row
            res = np.zeros((N,M), dtype=complex)

            for row_index in N:
                res[row_index, :] = DFT.fast_one_dimension(a[row_index, :])
            Q = a.reshape()
            q1  = DFT.fast_two_dimension(a[::2][::2])
            q2  = DFT.fast_one_dimension(a[1::2])
            
            multiplier = np.exp(-2j * np.pi * np.arange(N) / N)
            return np.concatenate([even + factor[:N / 2] * odd,
                                   even + factor[N / 2:] * odd])

    @staticmethod
    def test():
        # one dimension
        a = np.random.random(1024)
        fft = np.fft.fft(a)

        # two dimensions
        a2 = np.random.rand(32, 32)
        fft2 = np.fft.fft2(a2)

        tests = (
            (DFT.slow_one_dimension, a, fft),
            (DFT.slow_one_dimension_inverse, fft, a),
            (DFT.fast_one_dimension, a, fft),
            (DFT.fast_one_dimension_inverse, fft, a),
            (DFT.slow_two_dimension, a2, fft2),
            (DFT.slow_two_dimension_inverse, fft2, a2)
        )

        for method, args, expected in tests:
            if not np.allclose(method(args), expected):
                print(args)
                print(method(args))
                print(expected)
                raise AssertionError(
                    "{} failed the test".format(method.__name__))
