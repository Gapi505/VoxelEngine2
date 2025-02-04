use std::ops::Add;
use std::time::Instant;
use bevy::asset::io::ErasedAssetWriter;
use bevy::prelude::*;
use bevy::render::primitives::Aabb;
use bevy::tasks::{block_on, AsyncComputeTaskPool};
use bevy::tasks::futures_lite::{future, StreamExt};
use bevy::utils::futures;
use bevy_flycam::prelude::*;

use components::*;
mod components;



const RENDER_DISTANCE: i32 = 8;


fn main() {
    let mut app = App::new();
    app.add_plugins((DefaultPlugins, Setup));

    app.run();
}

struct Setup;

impl Plugin for Setup {
    fn build(&self, app: &mut App) {
        app.add_plugins(NoCameraPlayerPlugin)
            .insert_resource(KeyBindings{
                move_ascend: KeyCode::Space,
                move_descend: KeyCode::ShiftLeft,
                ..Default::default()
            })
            .insert_resource(MovementSettings{
                speed: 120.,
                ..Default::default()
            });

        app.insert_resource(Chunks::new());


        app.add_systems(Startup, (spawn_test_chunks, prepare_scene));
        app.add_systems(Update, (generate_chunks_async,poll_chunk_generations));
        app.add_systems(Update, (mesh_chunks_async,poll_chunk_meshing));
        // app.add_systems(Update, (update_neighbour_data));
        app.add_systems(Update, (spawn_chunks_around_player, destroy_chunks_away_from_player));
    }
}


#[derive(Component)]
struct Player;

fn prepare_scene(
    mut commands: Commands,
){
    commands.spawn((DirectionalLight{
        illuminance: light_consts::lux::AMBIENT_DAYLIGHT,
        shadows_enabled: true,
        ..Default::default()
    },
        Transform::from_rotation(Quat::from_rotation_x(-45f32.to_radians()).add(Quat::from_rotation_z(30f32.to_radians()))),
    ));

    commands.spawn((
        FlyCam,
        Camera3d{
            ..default()
        },
        Player
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

fn spawn_chunks_around_player(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut chunks: ResMut<Chunks>,
    player: Query<&Transform, With<Player>>,
){
    let trans = player.get_single().unwrap();
    let render_distance = RENDER_DISTANCE;
    for x in -render_distance..render_distance {
        for y in -render_distance..render_distance {
            for z in -render_distance..render_distance {
                let chunk_pos_offset = IVec3::new(x, y, z);
                let player_chunkpos = trans.translation.as_ivec3() / CHUNK_SIZE as i32;
                let chunkpos = player_chunkpos + chunk_pos_offset;
                if let None = chunks.get_raw(chunkpos) {
                    spawn_chunk(
                        &mut commands,
                        &mut chunks,
                        &mut materials,
                        &mut meshes,
                        ChunkPosition::from_ivec(chunkpos)
                    );
                }
            }
        }
    }
}

fn destroy_chunks_away_from_player(
    mut commands: Commands,
    mut chunks: ResMut<Chunks>,
    query: Query<(Entity, &Chunk), Without<ProcessingLock>>,
    player: Query<&Transform, With<Player>>
){

    let trans = player.get_single().unwrap();
    for (entity, chunk) in query.iter() {
        let chunkpos = chunk.chunk_position.as_ivec3();
        let player_chunkpos = trans.translation.as_ivec3() / CHUNK_SIZE as i32;
        let dst = player_chunkpos - chunkpos;
        if dst.x.abs() > RENDER_DISTANCE || dst.y.abs() > RENDER_DISTANCE || dst.z.abs() > RENDER_DISTANCE {
            commands.entity(entity).despawn();
            chunks.remove(ChunkPosition::from_ivec(chunkpos));
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
        //NeedsToUpdateNeighbours,
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
            let start_time = Instant::now(); // Record the start time
            let data = chunk_clone.generate();
            let duration = start_time.elapsed(); // Measure elapsed time
            println!("Generation completed in: {:?}", duration);
            data
        });
        commands.entity(entity).insert((ProcessingGeneration(task), ProcessingLock));
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
            if !chunk.is_empty(){
                commands.entity(entity).insert(NeedsMeshing);
            }
            else {
                commands.entity(entity).despawn();
            }
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
            let start_time = Instant::now(); // Record the start time
            let mesh = chunk_clone.cull_mesher();
            let duration = start_time.elapsed(); // Measure elapsed time
            println!("Meshing completed in: {:?}", duration);
            mesh
        });

        commands.entity(entity).insert((ProcessingMeshing(task)));
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
            commands.entity(entity).remove::<ProcessingLock>();
            //println!("meshed")
        }
    }
}

fn update_nearby_chunks(
    mut chunks_query: Query<&Chunk, With<NeedsToUpdateNeighbours>>,
    chunks: Res<Chunks>,
    mut connamds: Commands
) {
    for cur_chunk in chunks_query.iter_mut(){
        let offsets = Direction::iter();
        for offset in offsets{
            let pos = cur_chunk.chunk_position.as_ivec3() - offset.to_ivec();
            let chunk = chunks.get_raw(pos);

        }

    } 
}

#[derive(Component)]
struct NeedsToUpdateNeighbours;


#[derive(Component)]
struct ProcessingLock;

