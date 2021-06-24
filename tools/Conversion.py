import numpy as np

class Conversion:

    VOLTAGE_CHANNEL = 0
    CURRENT_CHANNEL = 1

    # rough assumption: 325 (V pp) covers ~65% of the dynamic range (15 bits because we have 16-bit unsigned int)
    # hence 0.65 * 2 ** 15
    DEFAULT_VOLTAGE_SCALING = 325 / (0.65 * 2 ** 15)

    # rough assumption: 18 (A pp) covers ~70% of the dynamic range (15 bits because we have 16-bit unsigned int)
    # hence 0.7 * 2 ** 15
    DEFAULT_CURRENT_SCALING = 18 / (0.7 * 2 ** 15)

    DEFAULT_SAMPLE_RATE = 10000  # Hz
    DEFAULT_MAINS_FREQ = 50  # Hz
    
    def __init__(self, data: np.ndarray):
        self.__voltage, self.__current = self.__prepare_date(data)
        
        self.real_power = self.__realPower()
        self.apparent_power = self.__apparentPower()
        self.reactive_power = self.__reactivePower()
        self.thd = self.__totalHarmonicDistortion()

    def __prepare_date(self, data: np.ndarray):
        recentered = data - data.mean(axis=0)
        recentered[:, self.VOLTAGE_CHANNEL] *= self.DEFAULT_VOLTAGE_SCALING
        recentered[:, self.CURRENT_CHANNEL] *= self.DEFAULT_CURRENT_SCALING
        self.voltage, self.current = recentered[:, self.VOLTAGE_CHANNEL], recentered[:, self.CURRENT_CHANNEL]

        # Fundental Frequency is 50Hz
        # 10000Hz / 50Hz = 200
        voltage = self.voltage.reshape((1500, 200))
        current = self.current.reshape((1500, 200))
        return voltage, current
    
    def __realPower(self):
        real_power = (self.__voltage * self.__current).mean(axis = 1)
        return real_power

    def __apparentPower(self):
        V = np.sqrt(np.mean(np.square(self.__voltage), axis = 1))
        A = np.sqrt(np.mean(np.square(self.__current), axis = 1))
        apparent_power = V * A
        return apparent_power

    def __reactivePower(self):
        reactive_power = np.sqrt(self.apparent_power ** 2 - self.real_power ** 2)
        return reactive_power

    def __totalHarmonicDistortion(self):
        cycles = self.__current

        # half + 1 of the double-sided frequency response
        fft_v = np.abs(np.fft.rfft(cycles)) / cycles.shape[-1]

        # single-sided response, so double all non-zeroth coefficients
        fft_v[:, 1:] *= 2

        # DC component is just the mean. numpy ellipsis (...) allows us to iterate over last axis without known rank.
        fft_v[..., 0] = np.mean(cycles)

        # read the fundamental and harmonic magnitudes from the appropriate slices
        w_fundamental = fft_v[..., 1]
        harmonic_magnitudes = fft_v[..., 2:]

        w_harmonics = np.sqrt(np.sum((np.power(harmonic_magnitudes, 2)), axis=-1))

        thd = w_harmonics / (w_fundamental + 1e-15)
        return thd