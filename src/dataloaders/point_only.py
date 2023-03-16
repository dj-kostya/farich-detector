from src.dataloaders import RootDataLoader


class PointOnlyDataloader(RootDataLoader):
    def __iter__(self):
        yield from self.genChunkFromRoot(event_chunk_size=1)
