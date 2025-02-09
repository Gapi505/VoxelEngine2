use std::ops::Add;
use std::time::Instant;
use bevy::asset::io::ErasedAssetWriter;
use bevy::prelude::*;
use bevy::render::primitives::Aabb;
use bevy::tasks::{block_on, AsyncComputeTaskPool};
use bevy::tasks::futures_lite::{future, StreamExt};
use bevy::utils::futures;
use bevy_flycam::prelude::*;
use bevy::pbr::{CascadeShadowConfigBuilder, NotShadowCaster};
use std::collections::VecDeque;

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
        app.insert_resource(ChunkLoadQueue{queue: VecDeque::new()});


        app.add_systems(Startup, (spawn_test_chunks, prepare_scene));
        app.add_systems(Update, (generate_chunks_async,poll_chunk_generations));
        app.add_systems(Update, (mesh_chunks_async,poll_chunk_meshing));
        // app.add_systems(Update, (update_neighbour_data));
        app.add_systems(Update, (spawn_chunks_around_player, spawn_queued_chunks, destroy_chunks_away_from_player, move_skybox));
    }
}


#[derive(Component)]
struct Player;

fn prepare_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>
){
    let far_distance = (RENDER_DISTANCE*CHUNK_SIZE as i32) as f32;

    let cascade_config = CascadeShadowConfigBuilder{
            first_cascade_far_bound: 90.,
            maximum_distance: far_distance,
            ..Default::default()
        }.build();
    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::AMBIENT_DAYLIGHT,
            shadows_enabled: true,
            ..Default::default()
        },
        Transform::from_rotation(
            Quat::from_rotation_x(-45f32.to_radians()) // Lower sun angle
                .mul_quat(Quat::from_rotation_y(30f32.to_radians())), // Slight tilt for natural shadow spread
        ),
        cascade_config
    ));

    commands.insert_resource(AmbientLight {
        brightness: 0.1, // Adjust this value (0.0 = no ambient, 1.0 = full bright)
        color: Color::srgb(0.8, 0.9, 1.0), // Slight blue tint to mimic sky reflection
    });
    commands.spawn((
        FlyCam,
        Camera3d::default(),
        DistanceFog{
            color: Color::srgb(0.35, 0.48, 0.66),
            directional_light_color: Color::srgba(1., 0.95, 0.85, 0.5),
            directional_light_exponent: 30.,
            falloff: FogFalloff::from_visibility_colors(
                far_distance,
                Color::srgb(0.35,0.5,0.66),
                Color::srgb(0.8,0.844,1.)
            )
        },
        Player
        ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.,1.,1.))),
        MeshMaterial3d(materials.add(StandardMaterial{
            base_color: Color::srgb(0.6,0.6,0.6),
            unlit: true,
            cull_mode: None,
            ..default()
        })),
        Transform::from_scale(Vec3::splat(far_distance*2.)),
        NotShadowCaster,
        SkyboxCube
    ));
}

#[derive(Component)]
struct SkyboxCube;
fn move_skybox(
    mut skybox: Query<&mut Transform, With<SkyboxCube>>,
    player: Query<&mut Transform, (With<Player>, Without<SkyboxCube>)>
){
    let mut transform = skybox.get_single_mut().unwrap();
    let player = player.get_single().unwrap();
    transform.translation = player.translation
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
    mut cqueue: ResMut<ChunkLoadQueue>,
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
                    if !cqueue.queue.contains(&chunkpos){
                        cqueue.queue.push_back(chunkpos)
                    }
                }
            }
        }
    }
}

fn spawn_queued_chunks(
    mut commands: Commands,

    mut cqueue: ResMut<ChunkLoadQueue>,
    time: Res<Time>,
    mut frame_timer: Local<f32>,

    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut chunks: ResMut<Chunks>,
){
    let max_spawns_per_frame = 500;
    for i in 0..max_spawns_per_frame{
        if let Some(pos) = cqueue.queue.pop_front(){
            spawn_chunk(
                &mut commands,
                &mut chunks,
                &mut materials,
                ChunkPosition::from_ivec(pos)
            )
        }
        else{
            break;
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

#[derive(Resource)]
struct ChunkLoadQueue{
    pub queue: VecDeque<IVec3>
}

fn spawn_chunk(
    mut commands: &mut Commands,
    mut chunks: &mut ResMut<Chunks>,
    mut materials: &mut ResMut<Assets<StandardMaterial>>,
    chunk_position: ChunkPosition
){
    let chunk = Chunk::new(&chunk_position);


    let chunk_ent = commands.spawn((
        chunk,
        Transform::from_translation(chunk_position.as_ivec3().as_vec3()*CHUNK_SIZE as f32),
        NeedsGeneration,
        // Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(materials.add(
            StandardMaterial{
                base_color: Color::WHITE,
                cull_mode: None,
                ..Default::default()
            }
        )),
        Aabb::from_min_max(Vec3::ZERO, Vec3::splat(CHUNK_SIZE as f32)),
        ProcessingLock,
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
        if let Some((data, neighbour_data, is_empty)) = block_on(future::poll_once(&mut gen_task.0)){
            commands.entity(entity).remove::<ProcessingGeneration>();
            if !is_empty{
                chunk.data = Some(data);
                chunk.neighbour_block_data = Some(neighbour_data);
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




#[derive(Component)]
struct ProcessingLock;

