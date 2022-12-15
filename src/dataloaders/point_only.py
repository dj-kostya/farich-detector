from src.dataloaders import RootDataLoader


class PointOnlyDataloader(RootDataLoader):
    def __iter__(self):
        for hit_df, part_df in self.genChunkFromRoot(event_chunk_size=1):
            yield hit_df
