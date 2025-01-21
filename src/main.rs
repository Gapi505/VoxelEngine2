use std::ops::Add;
use bevy::prelude::*;
use bevy::render::primitives::Aabb;
use bevy::tasks::{block_on, AsyncComputeTaskPool};
use bevy::tasks::futures_lite::{future, StreamExt};
use bevy::utils::futures;
use bevy_flycam::prelude::*;

use components::*;
mod components;
fn main() {
    let mut app = App::new();
    app.add_plugins((DefaultPlugins, Setup));

    app.run();
}

struct Setup;

impl Plugin for Setup {
    fn build(&self, app: &mut App) {
        app.add_plugins(PlayerPlugin)
            .insert_resource(KeyBindings{
                move_ascend: KeyCode::Space,
                move_descend: KeyCode::ShiftLeft,
                ..Default::default()
            })
            .insert_resource(MovementSettings{
                speed: 70.,
                ..Default::default()
            });

        app.insert_resource(Chunks::new());


        app.add_systems(Startup, (spawn_test_chunks, prepare_scene));
        app.add_systems(Update, (generate_chunks_async,poll_chunk_generations));
        app.add_systems(Update, (mesh_chunks_async,poll_chunk_meshing));
        // app.add_systems(Update, (update_neighbour_data));
    }
}


fn prepare_scene(
    mut commands: Commands,
){
    commands.spawn((DirectionalLight{
        illuminance: light_consts::lux::DIRECT_SUNLIGHT,
        shadows_enabled: true,
        ..Default::default()
    },
        Transform::from_rotation(Quat::from_rotation_x(-70f32.to_radians()).add(Quat::from_rotation_z(30f32.to_radians()))),
    ));
}


fn spawn_test_chunks(
    mut commands: Commands,
    mut chunks: ResMut<Chunks>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
){
    for x in -2..2{
        for z in -2..2{
            for y in -2..2{
                spawn_chunk(
                    &mut commands,
                    &mut chunks,
                    &mut materials,
                    &mut meshes,
                    ChunkPosition::new(x, y, z)
                )
            }
        }
    }
}

fn spawn_chunk(
    mut commands: &mut Commands,
    mut chunks: &mut ResMut<Chunks>,
    mut materials: &mut ResMut<Assets<StandardMaterial>>,
    mut meshes: &mut ResMut<Assets<Mesh>>,
    chunk_position: ChunkPosition
){
    let chunk = Chunk::new(&chunk_position);


    let chunk_ent = commands.spawn((
        chunk,
        Transform::from_translation(chunk_position.as_ivec3().as_vec3()*CHUNK_SIZE as f32),
        NeedsGeneration,
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(materials.add(
            StandardMaterial{
                base_color: Color::WHITE,
                ..Default::default()
            }
        )),
        Aabb::from_min_max(Vec3::ZERO, Vec3::splat(CHUNK_SIZE as f32)),
        NeedsToUpdateNeighbours,
    )).id();

    chunks.insert(chunk_position, chunk_ent);
}


fn generate_chunks_async(
    mut commands: Commands,
    chunk_ents: Query<(Entity, &Chunk), With<NeedsGeneration>>,
    chunks: Res<Chunks>,
) {
    let thread_pool = AsyncComputeTaskPool::get();

    for (entity, chunk) in chunk_ents.iter(){
        let chunk_clone = chunk.clone(); // Clone the chunk data
        let task = thread_pool.spawn(async move {
            let data = chunk_clone.generate();
            data
        });

        commands.entity(entity).insert(ProcessingGeneration(task));
        commands.entity(entity).remove::<NeedsGeneration>();
    }
}


fn poll_chunk_generations(
    mut commands: Commands,
    mut chunk_ents: Query<(Entity, &mut Chunk, &mut ProcessingGeneration)>
){
    for (entity, mut chunk, mut gen_task) in chunk_ents.iter_mut(){
        if let Some(data) = block_on(future::poll_once(&mut gen_task.0)){
            chunk.data = data;
            commands.entity(entity).remove::<ProcessingGeneration>();
            commands.entity(entity).insert(NeedsMeshing);
        }
    }
}

fn mesh_chunks_async(
    mut commands: Commands,
    chunk_ents: Query<(Entity, &Chunk), With<NeedsMeshing>>,
) {
    let thread_pool = AsyncComputeTaskPool::get();

    for (entity, chunk) in chunk_ents.iter(){
        let chunk_clone = chunk.clone(); // Clone the chunk data
        let task = thread_pool.spawn(async move {
            let mesh = chunk_clone.cull_mesher();
            mesh
        });

        commands.entity(entity).insert(ProcessingMeshing(task));
        commands.entity(entity).remove::<NeedsMeshing>();
    }
}

fn poll_chunk_meshing(
    mut commands: Commands,
    mut chunk_ents: Query<(Entity, &mut Chunk, &mut ProcessingMeshing)>,
    mut meshes: ResMut<Assets<Mesh>>,
){
    for (entity, mut chunk, mut gen_task) in chunk_ents.iter_mut(){
        if let Some(mesh) = block_on(future::poll_once(&mut gen_task.0)){
            let mesh_handle = meshes.add(mesh);
            commands.entity(entity).insert(Mesh3d(mesh_handle));
            commands.entity(entity).remove::<ProcessingMeshing>();
            println!("meshed")
        }
    }
}


fn update_neighbour_data(
    mut chunk_ents: Query<(&mut Chunk, Entity), With<NeedsToUpdateNeighbours>>,
    chunks: Res<Chunks>,
    chunks_query: Query<&Chunk>,
    mut commands: Commands,
) {
    for (mut chunk, entity) in chunk_ents.iter_mut() {
        for direction in Direction::iter() {
            // Get the neighbor chunk's position
            let offset = match direction {
                Direction::Left => IVec3::new(-1, 0, 0),
                Direction::Right => IVec3::new(1, 0, 0),
                Direction::Front => IVec3::new(0, 0, 1),
                Direction::Back => IVec3::new(0, 0, -1),
                Direction::Top => IVec3::new(0, 1, 0),
                Direction::Bottom => IVec3::new(0, -1, 0),
            };
            let neighbor_pos = chunk.chunk_position.as_ivec3() + offset;

            // Try to get the neighbor entity
            if let Some(neighbor_entity) = chunks.get_raw(neighbor_pos) {
                if let Ok(neighbor_chunk) = chunks_query.get(neighbor_entity) {
                    // Update the boundary data for this direction
                    for y in 0..CHUNK_SIZE {
                        for x in 0..CHUNK_SIZE {
                            let value = match direction {
                                Direction::Left => neighbor_chunk.get_raw(IVec3::new(0, y as i32, x as i32)),
                                Direction::Right => neighbor_chunk.get_raw(IVec3::new(CHUNK_SIZE as i32 - 1, y as i32, x as i32)),
                                Direction::Front => neighbor_chunk.get_raw(IVec3::new(x as i32, y as i32, CHUNK_SIZE as i32 - 1)),
                                Direction::Back => neighbor_chunk.get_raw(IVec3::new(x as i32, y as i32, 0)),
                                Direction::Top => neighbor_chunk.get_raw(IVec3::new(x as i32, CHUNK_SIZE as i32 - 1, y as i32)),
                                Direction::Bottom => neighbor_chunk.get_raw(IVec3::new(x as i32, 0, y as i32)),
                            };
                            chunk.neighbour_block_data.set(direction, x as i32, y as i32, value);
                        }
                    }
                }
            }
        }

        // Remove the `NeedsToUpdateNeighbours` component
        commands.entity(entity).remove::<NeedsToUpdateNeighbours>();
        commands.entity(entity).insert(NeedsGeneration);
    }
}
