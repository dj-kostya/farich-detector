from src.dataloaders import RootDataLoader


class PointOnlyDataloader(RootDataLoader):
    def __iter__(self):
        for hit_df, part_df in self.genChunkFromRoot(event_chunk_size=1):
            hit_df['x_c'], hit_df['y_c'] = zip(
                *hit_df[['x_c', 'y_c']].apply(lambda args: self._calculate_coordinates_in_pixel(*args), axis=1))
            yield hit_df
