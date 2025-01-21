use bevy::prelude::*;
use bevy::tasks::Task;
use bevy::utils::HashMap;

pub const CHUNK_SIZE: usize = 32;

#[derive(Resource)]
pub struct Chunks {
    chunks: HashMap<IVec3, Entity>,
}
impl Chunks {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
        }
    }
    pub fn get(&self, pos: ChunkPosition) -> Option<Entity> {
        self.chunks.get(&pos.to_ivec3()).copied()
    }

    pub fn insert(&mut self, pos: ChunkPosition, entity: Entity) {
        self.chunks.insert(pos.to_ivec3(), entity);
    }
}

#[derive(Component, Clone)]
pub struct Chunk {
    pub chunk_position: ChunkPosition,
    pub(crate) data: [u8; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE],
    neighbour_block_data: NeighbourBlockData,
}
impl Chunk {
    pub fn new(chunk_position: &ChunkPosition) -> Self {
        let data = [0; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE];
        let neighbour_block_data = NeighbourBlockData::new();
        Self {
            chunk_position: chunk_position.clone(),
            data,
            neighbour_block_data,
        }
    }
    const fn pos_to_index(&self, pos: IVec3) -> usize {
        (pos.y as usize * CHUNK_SIZE * CHUNK_SIZE) // Y (up) contributes the largest stride
            + (pos.z as usize * CHUNK_SIZE)       // Z (forward/backward) contributes row stride
            + pos.x as usize // X (left/right) is the column
    }
    const fn index_to_pos(&self, index: usize) -> IVec3 {
        let x = index % CHUNK_SIZE;
        let z = (index / CHUNK_SIZE) % CHUNK_SIZE;
        let y = index / (CHUNK_SIZE * CHUNK_SIZE);
        IVec3::new(x as i32, y as i32, z as i32)
    }

    fn get_raw(&self, pos: IVec3) -> u8 {
        self.data[self.pos_to_index(pos)]
    }

    fn get_neighboured(&self, pos: IVec3) -> u8 {
        // Check if the position is within the chunk bounds
        if (0..CHUNK_SIZE as i32).contains(&pos.x)
            && (0..CHUNK_SIZE as i32).contains(&pos.y)
            && (0..CHUNK_SIZE as i32).contains(&pos.z)
        {
            // Position is within the chunk, return raw block data
            self.get_raw(pos)
        } else {
            // Determine which neighbor's boundary data to fetch
            let direction = Direction::from_offset(pos);
            let local_pos = Self::wrap_to_local(pos);

            // Get data from the neighbor's boundary
            self.neighbour_block_data
                .get(direction, local_pos.x, local_pos.y)
        }
    }

    fn wrap_to_local(pos: IVec3) -> IVec3 {
        IVec3::new(
            (pos.x.rem_euclid(CHUNK_SIZE as i32)),
            (pos.y.rem_euclid(CHUNK_SIZE as i32)),
            (pos.z.rem_euclid(CHUNK_SIZE as i32)),
        )
    }


    pub fn generate(&self) -> [u8; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE] {
        let mut data = self.data;
        for i in 0..data.len() {
            let pos = self.index_to_pos(i);
            if pos.y < 9{
                data[i] = 1;
            }
        }
        data
    }
}

#[derive(Clone)]
struct NeighbourBlockData {
    data: [[u8; CHUNK_SIZE * CHUNK_SIZE]; 6],
}
impl NeighbourBlockData {
    fn new() -> Self {
        let data = [[0; CHUNK_SIZE * CHUNK_SIZE]; 6];
        NeighbourBlockData { data }
    }
    pub const fn get_index(direction: Direction) -> usize {
        match direction {
            Direction::Left => 0,
            Direction::Right => 1,
            Direction::Front => 2,
            Direction::Back => 3,
            Direction::Top => 4,
            Direction::Bottom => 5,
        }
    }
    fn get(&self, direction: Direction, x: i32, y: i32) -> u8 {
        assert!(
            x >= 0 && x < CHUNK_SIZE as i32,
            "neighbour chunk: x out of bounds"
        );
        assert!(
            y >= 0 && y < CHUNK_SIZE as i32,
            "neighbour chunk: y out of bounds"
        );
        let index = Self::get_index(direction);
        let plane_ind = (y * CHUNK_SIZE as i32 + x) as usize;
        self.data[index][plane_ind]
    }
    fn set(&mut self, direction: Direction, x: i32, y: i32, value: u8) {
        assert!(x >= 0 && x < CHUNK_SIZE as i32, "x out of bounds");
        assert!(y >= 0 && y < CHUNK_SIZE as i32, "y out of bounds");
        let index = Self::get_index(direction);
        let plane_ind = (y * CHUNK_SIZE as i32 + x) as usize;
        self.data[index][plane_ind] = value;
    }
}

enum Direction {
    Left,
    Right,
    Front,
    Back,
    Top,
    Bottom,
}
impl Direction {
    fn iter() -> impl Iterator<Item = Direction> {
        [
            Self::Left,
            Self::Right,
            Self::Front,
            Self::Back,
            Self::Top,
            Self::Bottom,
        ]
        .into_iter()
    }
    fn from_ivec(ivec: IVec3) -> Self {
        if ivec.x < 0 {
            Self::Left
        } else if ivec.x > 0 {
            Self::Right
        } else if ivec.z < 0 {
            Self::Front
        } else if ivec.z > 0 {
            Self::Back
        } else if ivec.y < 0 {
            Self::Top
        } else {
            Self::Bottom
        }
    }
    fn from_offset(pos: IVec3) -> Self {
        if pos.x < 0 {
            Direction::Left
        } else if pos.x >= CHUNK_SIZE as i32 {
            Direction::Right
        } else if pos.z < 0 {
            Direction::Back
        } else if pos.z >= CHUNK_SIZE as i32 {
            Direction::Front
        } else if pos.y < 0 {
            Direction::Bottom
        } else {
            Direction::Top
        }
    }
}

#[derive(Clone)]
pub struct ChunkPosition {
    position: IVec3,
}
impl ChunkPosition {
    pub fn from_world_pos(world_position: IVec3) -> Self {
        ChunkPosition {
            position: world_position / CHUNK_SIZE as i32,
        }
    }
    pub fn new(x: i32,y:i32,z:i32) -> Self {
        ChunkPosition { position: IVec3::new(x,y,z)}
    }
    pub fn to_ivec3(&self) -> IVec3 {
        self.position
    }
}


#[derive(Component)]
pub struct NeedsGeneration;
#[derive(Component)]
pub struct ProcessingGeneration(pub Task<[u8; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE]>);
