from pathlib import Path

import numpy as np
import pandas as pd
import uproot3 as uproot

from src.dataloaders import BaseDataloader


class RootDataLoader(BaseDataloader):
    root_path: Path

    # Данные о частице (для переименования и сохранения)
    _part_rename_map = {'m_hits': 'nhits',  # число срабатываний в событии
                        'm_pos_primary._0': 'x_p',  # X-координата вылета частицы в мм
                        'm_pos_primary._1': 'y_p',  # Y-координата вылета частицы в мм
                        'm_pos_primary._2': 'z_p',  # Z-координата вылета частицы в мм
                        'm_dir_primary._0': 'nx_p',  # X-компонента единичного вектора направления частицы
                        'm_dir_primary._1': 'ny_p',  # Y-компонента единичного вектора направления частицы
                        'm_dir_primary._2': 'nz_p',  # Z-компонента единичного вектора направления частицы
                        'm_beta_primary': 'beta',  # скорость частицы в единицах скорости света
                        'm_theta_primary': 'theta_p',  # полярный угол направления частицы в радианах
                        'm_phi_primary': 'phi_p',  # азимутальный угол направления частицы в радианах
                        'm_momentum_primary': 'momentum'  # импульс частицы в МэВ/c
                        }

    # Наблюдаемые данные о срабатываниях (для переименования и сохранения)
    _hit_rename_map = {'m_hits.m_photon_pos_chip._0': 'x_c',  # X-координата срабатывания в мм
                       'm_hits.m_photon_pos_chip._1': 'y_c',  # Y-координата срабатывания в мм
                       'm_hits.m_photon_pos_chip._2': 'z_c',  # Z-координата срабатывания в мм
                       'm_hits.m_photon_time': 't_c'  # время срабатывания в нс
                       }

    def __init__(self, root_path: Path or str, verbose: bool = True):
        super(RootDataLoader, self).__init__()
        self.root_path = Path(root_path)
        self.verbose = verbose
        idf = self._readInfoFromRoot()
        # Определения параметров фотодетектора для генерации темнового шума

        nFileEvents = idf.at[0, 'nevents']
        if self.verbose:
            print(f'Processing ROOT file {self.root_path} with {nFileEvents} simulated events...', flush=True)

    def _readInfoFromRoot(self) -> pd.DataFrame:
        '''
        Получение информации о моделировании из ROOT-файла в виде датафрейма формой (1, N), где N - число параметров моделирования.
        '''
        # Названия используемых колонок данных для переименования и сохранения в data frame
        idf_rename_map = {'m_num_events': 'nevents',  # число событий моделирования
                          'm_z_dis': 'zdis',  # расстояние от места рождения частицы до входа в радиатор в мм
                          'm_layers': 'nlayers',  # число слоев радиатора
                          'm_size': 'array_size',  # размер матрицы КФУ в мм
                          'm_gap': 'array_gap',  # зазор между матрицами КФУ в мм
                          'm_chip_size': 'pixel_size',  # размер пикселя КФУ в мм
                          'm_chip_pitch': 'pixel_gap',  # зазор между пикселями КФУ в мм
                          'm_chip_num_size': 'pixel_numx',  # размер матрицы КФУ в пикселях
                          'm_num_side_x': 'nxarrays', 'm_num_side_y': 'nyarrays',
                          # размер фотодетектора в матрицах КФУ по X и Y
                          'm_focal_length': 'distance',  # расстояние от входа в радиатор до входа в фотодетектор
                          'm_trg_window': 'trg_window_ns',  # размер временного окна в нс
                          'W': 'W',  # толщина радиатора в мм (вычисляемая)
                          'n_mean': 'n_mean',  # средний показатель преломления радиатора (вычисляемый)
                          'n_max': 'n_max',  # максимальный показатель преломления радиатора (вычисляемый)
                          }

        # Открытие ROOT-файла с данными используя Uproot https://github.com/scikit-hep/uproot3
        with uproot.open(self.root_path) as file:
            idf = file['info_sim'].pandas.df('*', flatten=False)

        # Переименование параметров
        idf.rename(columns=idf_rename_map, inplace=True, errors='ignore')

        # Получение параметров (многослойного) радиатора одинаковых для всех файлов
        n_l = idf.at[0, 'm_layers.first']  # показатели преломления слоёв
        w_l = idf.at[0, 'm_layers.second']  # толщины слоёв радиатора

        W = w_l.sum()  # суммарная толщина всех слоёв
        n_mean = n_l.mean()  # средний показатель преломления
        n_max = n_l.max()  # максимальный показатель преломления

        # Добавление вычисляемых параметров в idf
        idf['W'] = W
        idf['n_mean'] = n_mean
        idf['n_max'] = n_max

        # Сохранение нужных параметров
        idf = idf[idf_rename_map.values()]

        if self.verbose:
            for name in idf.columns:
                print(f'{name}: {idf.at[0, name]}')

        return idf

    def genChunkFromRoot(self, event_chunk_size=2000):
        '''
        Генератор событий из ROOT-файла в виде датафрейма. Число событий eventchunksize, читаемых генератором за один раз, должно выбираться так,
        чтобы все данные с учетом добавляемых шумовых срабатываний умещались в размер ОЗУ.

        Параметры:
          eventchunksize - число событий, загружаемых из ROOT-файла за один вызов.

        Описание условий моделирования:
          Ось Z направлена вдоль нормали к плоскости радиатора от радиатора к фотодетектору.
          Оси X и Y паралельны осям симметрии матрицы фотодетектора.
          Первичная частица (отрицательный пион) вылетает на расстоянии zdis=1 мм перед радиатором в его сторону
          Начальное положение частицы случайно разбрасывается по X и Y в квадрате со стороной (pixel_size+pixel_gap).
          Направление частицы случайно разбрасывается в телесном угле в пределах theta_p=[0, π/4], phi_p=[0, 2π].
          Скорость частицы случайно и равномерно разбрасывается от 0.957 до 0.999 скорости света.
        '''
        # Цикл чтения кусков ROOT-файла
        for partdf, hitdf in zip(
                uproot.pandas.iterate(str(self.root_path), "raw_data", self._part_rename_map.keys(),
                                      entrysteps=event_chunk_size),
                uproot.pandas.iterate(str(self.root_path), "raw_data", self._hit_rename_map.keys(),
                                      entrysteps=event_chunk_size,
                                      flatten=True)):
            if self.verbose:
                print('\n  Processing next chunk...')

            # Переименование колонок
            partdf.rename(columns=self._part_rename_map, inplace=True, errors='raise')
            hitdf.rename(columns=self._hit_rename_map, inplace=True, errors='raise')

            partdf = partdf.astype('float32', copy=False)
            partdf['nhits'] = partdf['nhits'].astype('int32', copy=False)
            hitdf = hitdf.astype('float32', copy=False)
            if self.verbose:
                print(f'    {hitdf.index.levels[0].size} entries with {hitdf.shape[0]} hits to process')

            yield hitdf, partdf

    def __iter__(self):
        # Слияние данных событий и срабатываний
        for hitdf, partdf in self.genChunkFromRoot(event_chunk_size=1):
            edf = hitdf.join(partdf, on='entry')
            yield edf
