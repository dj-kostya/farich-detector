from pathlib import Path

import uproot3 as uproot
import numpy as np
import pandas as pd
from typing import Iterator


def _show_uproot_tree(obj, max_key_len=12, sep='/', indent=0) -> None:
    width = max_key_len + len(sep)
    start_line = False
    if isinstance(obj, uproot.rootio.ROOTDirectory):
        print('TFile: ' + obj.name.decode('utf-8'))
        start_line = True
        indent = 2
    elif issubclass(type(obj), uproot.tree.TTreeMethods):
        print('TTree: ' + obj.name.decode('utf-8'))
        start_line = True
        indent = 4
    else:
        if len(obj.keys()) > 0:
            indent += width
            s = obj.name.decode('utf-8')[:max_key_len]
            print(s + ' ' * (max_key_len - len(s)) + sep, end='')
        else:
            print(obj.name.decode('utf-8'))

    if len(obj.keys()) > 0:
        for i, key in enumerate(obj.keys()):
            if i > 0 or start_line:
                print(' ' * indent, end='')
            _show_uproot_tree(obj[key], indent=indent)
        indent -= width


def show_uproot_tree(filepath: str or Path, max_key_len=12, sep='/', indent=0) -> None:
    with uproot.open(filepath) as f:
        _show_uproot_tree(f, max_key_len, sep, indent)


def readInfoFromRoot(filepath: str or Path, verbose: bool = True) -> pd.DataFrame:
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
    with uproot.open(filepath) as file:
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

    if verbose:
        for name in idf.columns:
            print(f'{name}: {idf.at[0, name]}')

    return idf


def genChunkFromRoot(filepath: str or Path, eventchunksize=2000, noisefreqpersqmm: float = 2e6,
                     noiseTimeRange: tuple = (0, 7),
                     shiftSignalTimes: bool = True,
                     edfstore: pd.HDFStore = None, verbose: bool = True, needNoise: bool = True) -> Iterator[pd.DataFrame]:
    '''
    Генератор событий из ROOT-файла в виде датафрейма. Число событий eventchunksize, читаемых генератором за один раз, должно выбираться так,
    чтобы все данные с учетом добавляемых шумовых срабатываний умещались в размер ОЗУ.

    Параметры:
      filepath - путь к ROOT-файлу для чтения.
      eventchunksize - число событий, загружаемых из ROOT-файла за один вызов.
      noisefreqpersqmm - частота темновых срабатываний на единицу активной площади фотодетектора в с^{-1}*мм^{-2}, подмешиваемых к событиям;
                       максимальное значение параметра, которое имеет смысл рассматривать -- 2e6.
      noiseTimeRange - (start, stop) -- tuple, задающий временной интервал генерации шума в наносекундах.
      shiftSignalTimes - если True, случайно смещать времена сигнальных срабатываний в пределах временного окна генерации шума.
      edfstore - HDF-хранилище для записи датафрейма "edf"; данные добавляются к уже записанным в хранилище.
      verbose - флаг отладочной печати.

    Описание условий моделирования:
      Ось Z направлена вдоль нормали к плоскости радиатора от радиатора к фотодетектору.
      Оси X и Y паралельны осям симметрии матрицы фотодетектора.
      Первичная частица (отрицательный пион) вылетает на расстоянии zdis=1 мм перед радиатором в его сторону
      Начальное положение частицы случайно разбрасывается по X и Y в квадрате со стороной (pixel_size+pixel_gap).
      Направление частицы случайно разбрасывается в телесном угле в пределах theta_p=[0, π/4], phi_p=[0, 2π].
      Скорость частицы случайно и равномерно разбрасывается от 0.957 до 0.999 скорости света.
    '''
    global rng

    # Данные о частице (для переименования и сохранения)
    part_rename_map = {'m_hits': 'nhits',  # число срабатываний в событии
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
    hit_rename_map = {'m_hits.m_photon_pos_chip._0': 'x_c',  # X-координата срабатывания в мм
                      'm_hits.m_photon_pos_chip._1': 'y_c',  # Y-координата срабатывания в мм
                      'm_hits.m_photon_pos_chip._2': 'z_c',  # Z-координата срабатывания в мм
                      'm_hits.m_photon_time': 't_c'  # время срабатывания в нс
                      }

    # Наименования колонок для сохранения в датафрейм
    edfcolstosave = list(part_rename_map.values()) + list(hit_rename_map.values())

    # Чтение параметров моделирования
    idf = readInfoFromRoot(filepath)

    # Определения параметров фотодетектора для генерации темнового шума
    pixel_size, pixel_gap = idf.at[0, 'pixel_size'], idf.at[0, 'pixel_gap']
    array_size, array_gap = idf.at[0, 'array_size'], idf.at[0, 'array_gap']
    nxpixels_arr = idf.at[0, 'pixel_numx']
    nxpixels_tot = idf.at[0, 'nxarrays'] * nxpixels_arr
    igrid = np.arange(nxpixels_tot // 2)
    xpnts = array_gap / 2 + (igrid // nxpixels_arr) * (array_size + array_gap) + (igrid % nxpixels_arr) * (
            pixel_size + pixel_gap) + pixel_size / 2
    xpnts = np.sort(np.append(-xpnts, xpnts)).astype('float32')
    xgrid, ygrid = np.meshgrid(xpnts, xpnts)
    xgrid = xgrid.reshape(xgrid.size)
    ygrid = ygrid.reshape(ygrid.size)

    def addNoise(partdf: pd.DataFrame, hitdf: pd.DataFrame) -> pd.DataFrame:
        '''
        Генерация темновых срабатываний темнового шума и добавление в датафрейм (без учета "мёртвого" времени пикселя).
        partdf - датафрейм для частиц
        hitdf - датафрейм для срабатываний
        '''
        assert (np.isclose(pixel_size * nxpixels_arr + pixel_gap * (nxpixels_arr - 1), array_size))
        nevents = partdf.shape[0]  # число событий

        # среднее число шумовых срабатываний на событие
        munoise = (noiseTimeRange[1] - noiseTimeRange[0]) * 1e-9 * noisefreqpersqmm * (pixel_size ** 2) * (
                nxpixels_tot ** 2)

        print(f'    Generate noise with DCR per mm^2 {noisefreqpersqmm}, mean number of hits per event: {munoise:.2f}.')

        noisehits = rng.poisson(munoise,
                                nevents)  # генерация массива числа шумовых срабатываний в событиях по пуассоновскому распределению
        Ndc = int(noisehits.sum())  # общее число шумовых срабатываний (скаляр)
        signalhits = partdf['nhits'].to_numpy()  # массив числа сигнальных срабатываний по событиям

        # случайное смещение сигнальных срабатываний в пределах временного окна генерации шума
        if shiftSignalTimes:
            hitdf['t_c'] += np.repeat(rng.uniform(0, noiseTimeRange[1] - 2, nevents), partdf['nhits'])

        hitdf['signal'] = np.ones(signalhits.sum(), bool)  # разметка сигнальных срабатываний значением 'signal' True
        if Ndc == 0:  # если нет шумовых срабатываний
            return hitdf  # возвращаем исходный датафрейм с добавлением колонки 'signal'

        ich = rng.choice(xgrid.size, Ndc)  # генерация случайных номеров сработавших каналов с возможным повтором
        xh = xgrid[ich]  # x-координата сработавших каналов
        yh = ygrid[ich]  # y-координата сработавших каналов
        zh = hitdf.at[(0, 0), 'z_c']  # z-координата срабатываний (скаляр)
        th = rng.uniform(noiseTimeRange[0], noiseTimeRange[1],
                         size=Ndc)  # генерация времён срабатываний по однородному распределению

        # нумерация шумовых срабатываний по событиям
        ievent = np.repeat(partdf.index, noisehits)  # массив номеров событий для записи в датафрейм
        ihit = np.zeros(Ndc, 'int64')  # инициализация массива номеров срабатываний для записи в датафрейм
        index = 0
        for i in range(nevents):
            ihit[index:index + noisehits[i]] = signalhits[i] + np.arange(noisehits[i])
            index += noisehits[i]

        # создание датафрейма с шумовыми срабатываниями того же формата, что hitdf
        noisedf = pd.DataFrame({'x_c': xh, 'y_c': yh, 'z_c': zh, 't_c': th, 'signal': np.zeros(Ndc, bool)},
                               index=pd.MultiIndex.from_arrays((ievent, ihit), names=('entry', 'subentry')))

        # TO DO: случайное смещение кольца в фотодетекторе (сдвиг координат сигнальных хитов).
        # Сложность с реализацией для неравномерной сетки пикселей, т.к. зазоры между матрицами больше зазоров между пикселями в матрице.

        # сливаем сигнальный и шумовой датафрейм и сортируем указатель событий и срабатываний
        hitdf2 = pd.concat((hitdf, noisedf), copy=False).sort_index(level=('entry', 'subentry'))

        # обновляем количества срабатываний в partdf, добавляя количества шумовых срабатываний по событиям
        partdf['nhits'] += noisehits

        return hitdf2

    nFileEvents = idf.at[0, 'nevents']
    print(f'Processing ROOT file {filepath} with {nFileEvents} simulated events...', flush=True)

    # Цикл чтения кусков ROOT-файла
    for partdf, hitdf in zip(
            uproot.pandas.iterate(str(filepath), "raw_data", part_rename_map.keys(), entrysteps=eventchunksize),
            uproot.pandas.iterate(str(filepath), "raw_data", hit_rename_map.keys(), entrysteps=eventchunksize,
                                  flatten=True)):
        print('\n  Processing next chunk...')

        # Переименование колонок
        partdf.rename(columns=part_rename_map, inplace=True, errors='raise')
        hitdf.rename(columns=hit_rename_map, inplace=True, errors='raise')

        partdf = partdf.astype('float32', copy=False)
        partdf['nhits'] = partdf['nhits'].astype('int32', copy=False)
        hitdf = hitdf.astype('float32', copy=False)
        if needNoise:
            # Генерация и добавление шумовых срабатываний
            hitdf = addNoise(partdf, hitdf)

        print(f'    {hitdf.index.levels[0].size} entries with {hitdf.shape[0]} hits to process')

        # Слияние данных событий и срабатываний
        edf = hitdf.join(partdf, on='entry')

        if verbose:
            print(edf)

        if edfstore is not None:
            print(f'    Saving edf chunk...')
            edfstore.put('edf', edf, format='table', append=True)

        yield edf
