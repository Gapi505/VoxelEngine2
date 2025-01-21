use bevy::prelude::*;
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
                speed: 20.,
                ..Default::default()
            });

        app.insert_resource(Chunks::new());


        app.add_systems(Startup, spawn_test_chunks);
        app.add_systems(Update, (generate_chunks_async,poll_chunk_generations));
    }
}


fn spawn_test_chunks(
    mut commands: Commands,
    mut chunks: ResMut<Chunks>,
){
    for x in -2..2{
        for z in -2..2{
            spawn_chunk(
                &mut commands,
                &mut chunks,
                ChunkPosition::new(x, 0, z)
            )
        }
    }
}

fn spawn_chunk(
    mut commands: &mut Commands,
    mut chunks: &mut ResMut<Chunks>,
    chunk_position: ChunkPosition
){
    let chunk = Chunk::new(&chunk_position);


    let chunk_ent = commands.spawn((
        chunk,
        Transform::from_translation(chunk_position.to_ivec3().as_vec3()*CHUNK_SIZE as f32),
        NeedsGeneration
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
        }
        commands.entity(entity).remove::<ProcessingGeneration>();
    }
}