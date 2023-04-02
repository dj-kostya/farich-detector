from pathlib import Path

import numpy as np
import pandas as pd

from src.dataloaders import RootDataLoader


class NoiseDataLoader(RootDataLoader):

    def __init__(self, root_path: Path or str,
                 verbose: bool = True,
                 noise_freq_per_sqmm: float = 2e6,
                 noise_time_range: tuple = (0, 7),
                 shift_signal_times: bool = True,
                 only_hits: bool = False):
        super().__init__(root_path, verbose=verbose)
        self.only_hits = only_hits
        self.shift_signal_times = shift_signal_times
        self.noise_time_range = noise_time_range
        self.noise_freq_per_sqmm = noise_freq_per_sqmm
        self.rng = np.random.default_rng(12345)
        self._get_noise_info()

    def _get_noise_info(self):
        idf = self._readInfoFromRoot()

        # Определения параметров фотодетектора для генерации темнового шума
        self.pixel_size, self.pixel_gap = idf.at[0, 'pixel_size'], idf.at[0, 'pixel_gap']
        self.array_size, self.array_gap = idf.at[0, 'array_size'], idf.at[0, 'array_gap']
        self.nxpixels_arr = idf.at[0, 'pixel_numx']
        self.nxpixels_tot = idf.at[0, 'nxarrays'] * self.nxpixels_arr
        igrid = np.arange(self.nxpixels_tot // 2)
        xpnts = self.array_gap / 2 + (igrid // self.nxpixels_arr) * (self.array_size + self.array_gap) + (
                igrid % self.nxpixels_arr) * (
                        self.pixel_size + self.pixel_gap) + self.pixel_size / 2
        xpnts = np.sort(np.append(-xpnts, xpnts)).astype('float32')
        xgrid, ygrid = np.meshgrid(xpnts, xpnts)
        self.xgrid = xgrid.reshape(xgrid.size)
        self.ygrid = ygrid.reshape(ygrid.size)

    def _add_noise(self, part_df: pd.DataFrame, hit_df: pd.DataFrame) -> pd.DataFrame:
        '''
                Генерация темновых срабатываний темнового шума и добавление в датафрейм (без учета "мёртвого" времени пикселя).
                partdf - датафрейм для частиц
                hitdf - датафрейм для срабатываний
                '''
        assert (
            np.isclose(self.pixel_size * self.nxpixels_arr + self.pixel_gap * (self.nxpixels_arr - 1), self.array_size))
        nevents = part_df.shape[0]  # число событий
        # nevents = 1
        # среднее число шумовых срабатываний на событие
        munoise = (self.noise_time_range[1] - self.noise_time_range[0]) * 1e-9 * self.noise_freq_per_sqmm * (
                self.pixel_size ** 2) * (
                          self.nxpixels_tot ** 2)
        if self.verbose:
            print(
                f'    Generate noise with DCR per mm^2 {self.noise_freq_per_sqmm}, mean number of hits per event: {munoise:.2f}.')

        noisehits = self.rng.poisson(munoise,
                                     nevents)  # генерация массива числа шумовых срабатываний в событиях по пуассоновскому распределению
        Ndc = int(noisehits.sum())  # общее число шумовых срабатываний (скаляр)
        signalhits = part_df['nhits'].to_numpy()  # массив числа сигнальных срабатываний по событиям

        # случайное смещение сигнальных срабатываний в пределах временного окна генерации шума
        if self.shift_signal_times:
            hit_df['t_c'] += np.repeat(self.rng.uniform(0, self.noise_time_range[1] - 2, nevents), part_df['nhits'])

        hit_df['signal'] = np.ones(signalhits.sum(), bool)  # разметка сигнальных срабатываний значением 'signal' True
        if Ndc == 0:  # если нет шумовых срабатываний
            return hit_df  # возвращаем исходный датафрейм с добавлением колонки 'signal'

        ich = self.rng.choice(self.xgrid.size,
                              Ndc)  # генерация случайных номеров сработавших каналов с возможным повтором
        xh = self.xgrid[ich]  # x-координата сработавших каналов
        yh = self.ygrid[ich]  # y-координата сработавших каналов
        zh = hit_df.at[hit_df.index[0], 'z_c']  # z-координата срабатываний (скаляр)
        th = self.rng.uniform(self.noise_time_range[0], self.noise_time_range[1],
                              size=Ndc)  # генерация времён срабатываний по однородному распределению

        # нумерация шумовых срабатываний по событиям
        ievent = np.repeat(part_df.index, noisehits)  # массив номеров событий для записи в датафрейм
        ihit = np.zeros(Ndc, 'int64')  # инициализация массива номеров срабатываний для записи в датафрейм
        index = 0
        for i in range(nevents):
            ihit[index:index + noisehits[i]] = signalhits[i] + np.arange(noisehits[i])
            index += noisehits[i]
        # создание датафрейма с шумовыми срабатываниями того же формата, что hitdf
        noisedf = pd.DataFrame({'x_c': xh, 'y_c': yh, 'z_c': zh, 't_c': th, 'signal': np.zeros(Ndc, bool)},
                               index=pd.MultiIndex.from_arrays((ievent, ihit), names=('entry', 'subentry')))

        # TO DO: случайное смещение кольца в фотодетекторе (сдвиг координат сигнальных хитов).
        # Сложность с реализацией для неравномерной сетки пикселей,
        # т.к. зазоры между матрицами больше зазоров между пикселями в матрице.
        if not noisedf.empty:
            noisedf['x_c'], noisedf['y_c'] = zip(
                *noisedf[['x_c', 'y_c']].apply(lambda args: self._calculate_coordinates_in_pixel(*args), axis=1))
        # сливаем сигнальный и шумовой датафрейм и сортируем указатель событий и срабатываний
        hitdf2 = pd.concat((hit_df, noisedf), copy=False).sort_index(level=('entry', 'subentry'))

        # обновляем количества срабатываний в partdf, добавляя количества шумовых срабатываний по событиям
        part_df['nhits'] += noisehits

        return hitdf2

    def genChunkFromRoot(self, event_chunk_size=2000):
        for hit_df, part_df in super().genChunkFromRoot(event_chunk_size=event_chunk_size):
            hit_df = self._add_noise(part_df, hit_df)
            if self.only_hits:
                yield hit_df
            else:
                yield hit_df.join(part_df, on='entry')

    def __iter__(self):
        yield from self.genChunkFromRoot(event_chunk_size=1)