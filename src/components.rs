use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::tasks::Task;
use bevy::utils::HashMap;
use noise::{NoiseFn, Perlin};
use fastnoise_lite::{NoiseType, FastNoiseLite};

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
        self.chunks.get(&pos.as_ivec3()).copied()
    }
    pub fn get_raw(&self, pos: IVec3) -> Option<Entity> {
        self.chunks.get(&pos).copied()
    }

    pub fn insert(&mut self, pos: ChunkPosition, entity: Entity) {
        self.chunks.insert(pos.as_ivec3(), entity);
    }
    pub fn remove(&mut self, pos: ChunkPosition) {
        self.chunks.remove(&pos.as_ivec3());
    }
}

#[derive(Component, Clone)]
pub struct Chunk {
    pub chunk_position: ChunkPosition,
    pub data: Option<[u8; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE]>,
    pub neighbour_block_data: Option<NeighbourBlockData>,
    pub seed: u64
}
impl Chunk {
    pub fn new(chunk_position: &ChunkPosition) -> Self {
        Self {
            chunk_position: chunk_position.clone(),
            data: None,
            neighbour_block_data: None,
            seed: 6782137862
        }
    }
    const fn pos_to_index(pos: IVec3) -> usize {
        pos.z as usize * CHUNK_SIZE * CHUNK_SIZE + pos.y as usize * CHUNK_SIZE + pos.x as usize
    }

    const fn index_to_pos(&self, index: usize) -> IVec3 {
        let x = index % CHUNK_SIZE;
        let y = (index / CHUNK_SIZE) % CHUNK_SIZE;
        let z = index / (CHUNK_SIZE * CHUNK_SIZE);
        IVec3::new(x as i32, y as i32, z as i32)
    }

    pub fn get_raw(data: &[u8; CHUNK_SIZE.pow(3)], pos: IVec3) -> u8 {
        data[Self::pos_to_index(pos)]
    }

    fn get_neighboured(data: &[u8; CHUNK_SIZE.pow(3)], neighbours: &NeighbourBlockData,pos: IVec3) -> u8 {
        // Check if the position is within the chunk bounds
        if (0..CHUNK_SIZE as i32).contains(&pos.x)
            && (0..CHUNK_SIZE as i32).contains(&pos.y)
            && (0..CHUNK_SIZE as i32).contains(&pos.z)
        {
            // Position is within the chunk, return raw block data
            Self::get_raw(&data, pos)

        } else {
            // Determine which neighbor's boundary data to fetch
            let direction = Direction::from_offset(pos);
            let local_pos = Self::wrap_to_boundary_2d(pos, &direction);

            // Get data from the neighbor's boundary
            neighbours.get(direction, local_pos.x, local_pos.y)
        }
    }

    fn wrap_to_local(pos: IVec3) -> IVec3 {
        IVec3::new(
            (pos.x.rem_euclid(CHUNK_SIZE as i32)),
            (pos.y.rem_euclid(CHUNK_SIZE as i32)),
            (pos.z.rem_euclid(CHUNK_SIZE as i32)),
        )
    }
    pub fn wrap_to_boundary_2d(pos: IVec3, direction: &Direction) -> IVec2 {
        match direction {
            Direction::Top | Direction::Bottom => IVec2::new(
                pos.x.rem_euclid(CHUNK_SIZE as i32),
                pos.z.rem_euclid(CHUNK_SIZE as i32),
            ),
            Direction::Left | Direction::Right => IVec2::new(
                pos.y.rem_euclid(CHUNK_SIZE as i32),
                pos.z.rem_euclid(CHUNK_SIZE as i32),
            ),
            Direction::Front | Direction::Back => IVec2::new(
                pos.x.rem_euclid(CHUNK_SIZE as i32),
                pos.y.rem_euclid(CHUNK_SIZE as i32),
            ),
        }
    }



    pub fn generate(&self) -> ([u8; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE],NeighbourBlockData, bool) {
        let mut data = [0; CHUNK_SIZE.pow(3)];
        let mut neighbour_data = NeighbourBlockData::new();
        let mut noisegen = FastNoiseLite::new();
        noisegen.set_noise_type(Some(NoiseType::Perlin));

        let mut celular_noise = FastNoiseLite::new();
        celular_noise.set_noise_type(Some(NoiseType::Cellular));
        let mut block_counter = 0;


        for x in -1..(CHUNK_SIZE+1) as i32{
            for z in -1..(CHUNK_SIZE+1) as i32{
                let pos2d = Vec2::new(x as f32, z as f32);
                /*let mut noise = self.noise_2d(&mut noisegen, 2., pos2d, 9182312)*300.-30.;
                noise += self.noise_2d(&mut noisegen, 1., pos2d, 91872378912)*30.-30.;
                noise += self.noise_2d(&mut noisegen, 0.2,pos2d, 2131231)*10.-5.;
                noise *= self.noise_2d(&mut noisegen, 8., pos2d, 8769132)*1.;
                noise += self.noise_2d(&mut noisegen, 3., pos2d, 1047638)*7.;
                noise += self.noise_2d(&mut noisegen,0.1 ,pos2d, 2917837893)*5.;
                noise += -smoothstep(0.1, 0.3, self.noise_2d(&mut celular_noise, 1., pos2d, 987321).powi(2))* 100.;
                */

                // --- BASE TERRAIN (Remove aggressive small-scale noise) ---
                let mut noise = self.noise_2d(&mut noisegen, 2., pos2d, 9182312) * 300. - 30.;
                noise += self.noise_2d(&mut noisegen, 1., pos2d, 91872378912) * 30. - 30.;
                noise += self.noise_2d(&mut noisegen, 0.6, pos2d, 2131231) * 10. - 5.;
                noise += self.noise_2d(&mut noisegen, 3., pos2d, 1047638) * 7.;
                noise += self.noise_2d(&mut noisegen, 0.1, pos2d, 2917837893) * 5.;

                // --- SOFT PERTURBATION (Much lower intensity) ---
                let perturb = self.noise_2d(&mut noisegen, 4., pos2d, 8769132) * 5. - 2.5; // Less extreme
                noise += perturb;

                // --- LARGE-SCALE RAVINE MASK (Ensures only some areas get deep ravines) ---
                let ravine_mask = self.noise_2d(&mut noisegen, 10., pos2d, 5678123); // Lower frequency for large-scale regions
                let deep_ravine_factor = smoothstep(0.4, 5., ravine_mask); // Controls deep ravine locations

                // --- CELLULAR NOISE FOR RAVINES ---
                let ravine_base = self.noise_2d(&mut celular_noise, 1., pos2d, 987321);
                let ravine_depth = smoothstep(0.25, 0.5, 1.0 - ravine_base) * (40. + deep_ravine_factor * 80.); // Lower max depth

                // --- RIDGE ENHANCEMENT (Softer and only in key places) ---
                let ridge_factor = (1.0 - ravine_base).powi(2) * 20. * deep_ravine_factor; // Lower ridge intensity

                // --- APPLY RAVINES WITH SMOOTHER BLENDING ---
                noise = mix(noise, noise - ravine_depth, deep_ravine_factor * 0.8); // Less extreme ravine effect
                noise += ridge_factor;
                 

                for y in -1..(CHUNK_SIZE+1) as i32 {
                    let pos = IVec3::new(x as i32,y as i32,z as i32);
                    if ((pos.y + self.chunk_position.as_ivec3().y * CHUNK_SIZE as i32) as f32)< noise{
                        //println!("here");
                        block_counter += 1;

                        let mut count = 0;
                        if x < 0 || x >= CHUNK_SIZE as i32 { count += 1; }
                        if z < 0 || z >= CHUNK_SIZE as i32 { count += 1; }
                        if y < 0 || y >= CHUNK_SIZE as i32 { count += 1; }
                        match count{
                            0 =>{
                                let i = Self::pos_to_index(pos);
                                data[i] = 1;

                            },
                            1 =>{
                                let dir = Direction::from_offset(pos);
                                let wrapped = Self::wrap_to_boundary_2d(pos, &dir);
                                neighbour_data.set(dir, wrapped.x, wrapped.y, 1);
                            },
                            _ => {}
                        }
                    }
                }
            }
        }
            
        let is_empty = if block_counter == 0 || block_counter == (CHUNK_SIZE+2).pow(3){true} else {false};
        (data, neighbour_data, is_empty)
    }
    fn noise_2d(&self, noise: &mut FastNoiseLite,  scale: f64, inchunk_position: Vec2, seed_offset: u64) -> f32{
        noise.set_seed(Some((self.seed + seed_offset) as i32));
        let global_x = (self.chunk_position.as_ivec3().x * CHUNK_SIZE as i32) as f32 + inchunk_position.x;
        let scaled_x = global_x as f64 / scale;
        let global_z = (self.chunk_position.as_ivec3().z * CHUNK_SIZE as i32) as f32 + inchunk_position.y;
        let scaled_z = global_z as f64 / scale;
        (noise.get_noise_2d(scaled_x as f32, scaled_z as f32)+1.)*0.5
    }


    pub fn cull_mesher(&self) -> Mesh {
        if let (Some(ref chunk_data), Some(nbd)) = (self.data, self.neighbour_block_data.clone()){

        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all());
        /*mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION,
                              vec![
                                  [0., 0., 0.], // Vertex 0
                                  [1., 0., 0.],  // Vertex 1
                                  [0., 0., 1.],   // Vertex 2
                                  ],);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![
            [0.,1.,0.],
            [0.,1.,0.],
            [0.,1.,0.],
        ]);
        mesh.insert_indices((Indices::U32(vec![
            0, 2, 1 // Indices for the single triangle
        ])));*/

        let mut ind_positions = Vec::new(); // Estimate for 4 vertices per block
        let mut normals = Vec::new();
        let mut indices = Vec::new(); // Estimate for 6 indices per block

        macro_rules! ind_vec {
            ($ind:expr, $flipped:expr) => {
                if $flipped {
                    vec![$ind + 0, $ind + 2, $ind + 1, $ind + 1, $ind + 2, $ind + 3]
                } else {
                    vec![$ind + 0, $ind + 1, $ind + 2, $ind + 1, $ind + 3, $ind + 2]
                }
            };
        }
        let mut top_norm = vec![
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 1., 0.]];
        let mut bottom_norm = vec![
            [0., -1., 0.],
            [0., -1., 0.],
            [0., -1., 0.],
            [0., -1., 0.]];
        let mut left_norm = vec![
            [-1., 0., 0.],
            [-1., 0., 0.],
            [-1., 0., 0.],
            [-1., 0., 0.]];
        let mut right_norm = vec![
            [1., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.]];
        let mut front_norm = vec![
            [0., 0., 1.],
            [0., 0., 1.],
            [0., 0., 1.],
            [0., 0., 1.]];
        let mut back_norm = vec![
            [0., 0., -1.],
            [0., 0., -1.],
            [0., 0., -1.],
            [0., 0., -1.]];

        let mut ind = 0;
        let mut i = 0;

        for x in 0..(CHUNK_SIZE) as i32{
            for z in 0..(CHUNK_SIZE) as i32{
                for y in 0..(CHUNK_SIZE) as i32{
                    let ipos = IVec3::new(x,y,z);
                    let i = Self::pos_to_index(ipos);
                    let (x, y, z) = (ipos.x as f32, ipos.y as f32, ipos.z as f32);
                    let block = chunk_data[i];
                    if block != 0 {
                        let mut top_face = vec![
                            [x, y + 1., z],
                            [x + 1., y + 1., z],
                            [x, y + 1., z + 1.],
                            [x + 1., y + 1., z + 1.]];
                        let mut bottom_face = vec![
                            [x, y, z],
                            [x + 1., y, z],
                            [x, y, z + 1.],
                            [x + 1., y, z + 1.]];
                        let mut left_face = vec![
                            [x, y, z],
                            [x, y + 1., z],
                            [x, y, z + 1.],
                            [x, y + 1., z + 1.]];
                        let mut right_face = vec![
                            [x + 1., y, z],
                            [x + 1., y + 1., z],
                            [x + 1., y, z + 1.],
                            [x + 1., y + 1., z + 1.]];
                        let mut front_face = vec![
                            [x, y, z + 1.],
                            [x, y + 1., z + 1.],
                            [x + 1., y, z + 1.],
                            [x + 1., y + 1., z + 1.]];
                        let mut back_face = vec![
                            [x, y, z],
                            [x, y + 1., z],
                            [x + 1., y, z],
                            [x + 1., y + 1., z]];
        
                        let mut neighbors = [
                            (Direction::Top.to_ivec(), &mut top_face, &mut top_norm, true),
                            (Direction::Bottom.to_ivec(), &mut bottom_face, &mut bottom_norm, false),
                            (Direction::Left.to_ivec(), &mut left_face, &mut left_norm, true),
                            (Direction::Right.to_ivec(), &mut right_face, &mut right_norm, false),
                            (Direction::Front.to_ivec(), &mut front_face, &mut front_norm, true),
                            (Direction::Back.to_ivec(), &mut back_face, &mut back_norm, false),
                        ];
        
                        for (offset, face, norm, flipped) in neighbors.iter_mut() {
                            if Self::get_neighboured(&chunk_data, &nbd,ipos + *offset) == 0 {
                                ind_positions.append(face);
                                normals.append(&mut norm.clone());
                                indices.append(&mut ind_vec!(ind, *flipped));
                                ind += 4;
                            }
                        }
        
                    }

                }
            }
        }
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, ind_positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.insert_indices(Indices::U32(indices));
        mesh
    }
        else{
            panic!("no data")
        }
    }

    //pub fn is_empty(&self) -> bool{
    //    self.data.iter().all(|b| b == &0) || self.data.iter().all(|b| b >= &1)
    //}
}

#[derive(Clone)]
pub struct NeighbourBlockData {
    data: [[u8; CHUNK_SIZE * CHUNK_SIZE]; 6],
}
impl NeighbourBlockData {
    fn new() -> Self {
        let data = [[0u8; CHUNK_SIZE * CHUNK_SIZE]; 6];
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
    pub fn get(&self, direction: Direction, x: i32, y: i32) -> u8 {
        let index = Self::get_index(direction);
        let plane_ind = (y * CHUNK_SIZE as i32 + x) as usize;
        self.data[index][plane_ind]
    }
    pub fn set(&mut self, direction: Direction, x: i32, y: i32, value: u8) {
        let index = Self::get_index(direction);
        let plane_ind = (y * CHUNK_SIZE as i32 + x) as usize;
        self.data[index][plane_ind] = value;
    }
}

#[derive(Clone, Copy)]
pub enum Direction {
    Left,
    Right,
    Front,
    Back,
    Top,
    Bottom,
}
impl Direction {
    pub fn iter() -> impl Iterator<Item = Direction> {
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
    pub fn from_ivec(ivec: IVec3) -> Self {
        if ivec.x < 0 {
            Self::Left
        } else if ivec.x > 0 {
            Self::Right
        } else if ivec.z < 0 {
            Self::Back
        } else if ivec.z > 0 {
            Self::Front
        } else if ivec.y < 0 {
            Self::Bottom
        } else {
            Self::Top
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

    pub const fn to_ivec(&self) -> IVec3 {
        match self {
            Direction::Left => IVec3::new(-1, 0, 0),
            Direction::Right => IVec3::new(1, 0, 0),
            Direction::Front => IVec3::new(0, 0, 1),
            Direction::Back => IVec3::new(0, 0, -1),
            Direction::Top => IVec3::new(0, 1, 0),
            Direction::Bottom => IVec3::new(0, -1, 0),
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
    pub fn from_ivec(ivec: IVec3) -> Self {
        Self{
            position: ivec,
        }
    }
    pub fn new(x: i32,y:i32,z:i32) -> Self {
        ChunkPosition { position: IVec3::new(x,y,z)}
    }
    pub fn as_ivec3(&self) -> IVec3 {
        self.position
    }
}


#[derive(Component)]
pub struct NeedsGeneration;
#[derive(Component)]
pub struct ProcessingGeneration(pub Task<([u8; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE], NeighbourBlockData, bool)>);

#[derive(Component)]
pub struct NeedsMeshing;
#[derive(Component)]
pub struct ProcessingMeshing(pub Task<Mesh>);

#[derive(Component)]
pub struct NeedsToUpdateNeighbours;


fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}


fn mix(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}
