use std::collections::VecDeque;
use std::time::Instant;

use bevy::pbr::{CascadeShadowConfigBuilder, NotShadowCaster};
use bevy::prelude::*;
use bevy::render::primitives::Aabb;
use bevy::tasks::{block_on, AsyncComputeTaskPool};
use bevy::tasks::futures_lite::{future, StreamExt};
use bevy::render::{
        extract_component::{
            ComponentUniforms, DynamicUniformIndex, ExtractComponent, ExtractComponentPlugin,
            UniformComponentPlugin,
        },
        render_graph::{
            NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::{
            binding_types::{sampler, texture_2d, uniform_buffer},
            *,
        },
        renderer::{RenderContext, RenderDevice},
        view::ViewTarget,
        RenderApp,
        render_asset::RenderAssetUsages,
    };
use bevy::ecs::query::QueryItem;
use bevy::core_pipeline::{
    core_3d::graph::{Core3d,Node3d},
    fullscreen_vertex_shader::fullscreen_shader_vertex_state
};
use bevy::ui::graph::NodeUi;

use rayon::prelude::*;

use avian3d::prelude::*;

mod components;
use components::*;


const RENDER_DISTANCE: i32 = 10;
const SHADER_ASSET_PATH: &str = "shaders/post.wgsl";


fn main() {
    let mut app = App::new();
    app.add_plugins((DefaultPlugins, Setup, PostProcessPlugin));

    app.run();
}

struct Setup;

impl Plugin for Setup {
    fn build(&self, app: &mut App) {
        app.add_plugins(PhysicsPlugins::default());
    
        app.insert_resource(Chunks::new());
        app.insert_resource(ChunkLoadQueue{queue: VecDeque::new()});


        app.add_systems(Startup, (prepare_scene, preload_assets, spawn_player, spawn_ui));
        app.add_systems(Update, (generate_chunks_async,poll_chunk_generations));
        app.add_systems(Update, (mesh_chunks_async,poll_chunk_meshing));
        app.add_systems(Update, (construct_colliders_async,poll_colliders));
        // app.add_systems(Update, (update_neighbour_data));
        app.add_systems(Update, (spawn_chunks_around_player, spawn_queued_chunks, destroy_chunks_away_from_player, move_skybox, handle_nearby_colliders));

        app.add_systems(FixedUpdate, (drone_controller, move_drone_camera));
    }
}

struct PostProcessPlugin;
impl Plugin for PostProcessPlugin{
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractComponentPlugin::<PostProcessSettings>::default(),
            UniformComponentPlugin::<PostProcessSettings>::default(),
        ));
        app.add_systems(Update, adjust_noise_shader);
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {return;};
        render_app
            .add_render_graph_node::<ViewNodeRunner<PostProcessNode>>(
                Core3d,
                PostProcessLabel,
            )
            .add_render_graph_edges(
                Core3d,
                (
                    //Node3d::Tonemapping,
                    NodeUi::UiPass,
                    PostProcessLabel,
                    //Node3d::EndMainPassPostProcessing,
                    Node3d::Upscaling,
                ),
            );
    }
    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {return;};
        render_app
            .init_resource::<PostProcessPipeline>();
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct PostProcessLabel;
#[derive(Default)]
struct PostProcessNode;

impl ViewNode for PostProcessNode{
    type ViewQuery = (
        &'static ViewTarget,
        &'static PostProcessSettings,
        &'static DynamicUniformIndex<PostProcessSettings>,
    );
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_target, _post_process_settings, settings_index): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError>{
        let post_process_pipeline =  world.resource::<PostProcessPipeline>();
        let pipeline_chace = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_chace.get_render_pipeline(post_process_pipeline.pipeline_id) else {return Ok(());};
        let settings_uniforms = world.resource::<ComponentUniforms<PostProcessSettings>>();
        let Some(settings_binding) = settings_uniforms.uniforms().binding() else {return Ok(());};

        let post_process = view_target.post_process_write();

        let bind_group = render_context.render_device().create_bind_group(
            "post_process_bind_group",
            &post_process_pipeline.layout,
            &BindGroupEntries::sequential((
                post_process.source,
                &post_process_pipeline.sampler,
                settings_binding.clone(),
            )),
        );

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor{
            label: Some("post_process_pass"),
            color_attachments: &[Some(RenderPassColorAttachment{
                view: post_process.destination,
                resolve_target: None,
                ops: Operations::default()
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[settings_index.index()]);
        render_pass.draw(0..3,0..1);
        Ok(())

    }
}



// This contains global data used by the render pipeline. This will be created once on startup.
#[derive(Resource)]
struct PostProcessPipeline {
    layout: BindGroupLayout,
    sampler: Sampler,
    pipeline_id: CachedRenderPipelineId,
}

impl FromWorld for PostProcessPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        // We need to define the bind group layout used for our pipeline
        let layout = render_device.create_bind_group_layout(
            "post_process_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                // The layout entries will only be visible in the fragment stage
                ShaderStages::FRAGMENT,
                (
                    // The screen texture
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    // The sampler that will be used to sample the screen texture
                    sampler(SamplerBindingType::Filtering),
                    // The settings uniform that will control the effect
                    uniform_buffer::<PostProcessSettings>(true),
                ),
            ),
        );

        // We can create the sampler here since it won't change at runtime and doesn't depend on the view
        let sampler = render_device.create_sampler(&SamplerDescriptor::default());

        // Get the shader handle
        let shader = world.load_asset(SHADER_ASSET_PATH);

        let pipeline_id = world
            .resource_mut::<PipelineCache>()
            // This will add the pipeline to the cache and queue its creation
            .queue_render_pipeline(RenderPipelineDescriptor {
                label: Some("post_process_pipeline".into()),
                layout: vec![layout.clone()],
                // This will setup a fullscreen triangle for the vertex state
                vertex: fullscreen_shader_vertex_state(),
                fragment: Some(FragmentState {
                    shader,
                    shader_defs: vec![],
                    // Make sure this matches the entry point of your shader.
                    // It can be anything as long as it matches here and in the shader.
                    entry_point: "fragment".into(),
                    targets: vec![Some(ColorTargetState {
                        format: TextureFormat::bevy_default(),
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                // All of the following properties are not important for this effect so just use the default values.
                // This struct doesn't have the Default trait implemented because not all fields can have a default value.
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                push_constant_ranges: vec![],
                zero_initialize_workgroup_memory: false,
            });

        Self {
            layout,
            sampler,
            pipeline_id,
        }
    }
}

// This is the component that will get passed to the shader
#[derive(Component, Default, Clone, Copy, ExtractComponent, ShaderType)]
struct PostProcessSettings {
    noise_strenght: f32,
    frame: f32,
    // WebGL2 structs must be 16 byte aligned.
    #[cfg(feature = "webgl2")]
    _webgl2_padding: Vec3,
}

fn adjust_noise_shader(
    mut settings: Query<&mut PostProcessSettings>,
    drone: Query<(&LinearVelocity, &DroneController)>,
    time: Res<Time>,
){
    let dt = time.delta_secs();
    let (speed, controller) = drone.get_single().unwrap();
    for mut setting in &mut settings{
        setting.noise_strenght = 0.3f32.lerp(0.95, speed.length().clamp(0., controller.top_speed)/controller.top_speed);
        setting.frame += dt*60.;
    }
}


#[derive(Component)]
struct Player;

fn prepare_scene(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>
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
        Mesh3d(meshes.add(Cuboid::new(1.,1.,1.))),
        MeshMaterial3d(materials.add(StandardMaterial{
            base_color: Color::srgb(0.6,0.6,0.6),
            unlit: true,
            cull_mode: None,
            ..default()
        })),
        Transform::from_scale(Vec3::splat(far_distance*2.)),
        NotShadowCaster,
        SkyboxCube,
        IsDefaultUiCamera,
    ));
}


fn spawn_ui(
    mut commands: Commands,
){
    commands
        .spawn(
            Node{
                width: Val::Percent(100.),
                height: Val::Percent(100.),
                flex_grow: 100.,
                justify_content: JustifyContent::Center,
                ..default()
            },
        )
        .with_children(|parent|{
            parent.spawn((
                Node{
                    position_type: PositionType::Relative,
                    align_self: AlignSelf::Center,
                    ..default()
                },
            )).with_children(|parent|{
                    parent.spawn((
                        Node{
                            width: Val::Vh(0.5),
                            height: Val::Vh(2.),
                            position_type: PositionType::Absolute,
                            bottom: Val::Vh(1.25),
                            right: Val::Vh(0.25),
                            ..default()
                        },
                        BackgroundColor(Color::WHITE),
                    ));
                    parent.spawn((
                        Node{
                            width: Val::Vh(0.5),
                            height: Val::Vh(2.),
                            position_type: PositionType::Absolute,
                            top: Val::Vh(0.25),
                            right: Val::Vh(0.25),
                            ..default()
                        },
                        BackgroundColor(Color::WHITE),
                    ));
                    parent.spawn((
                        Node{
                            width: Val::Vh(2.),
                            height: Val::Vh(0.5),
                            position_type: PositionType::Absolute,
                            bottom: Val::Vh(0.25),
                            right: Val::Vh(1.25),
                            ..default()
                        },
                        BackgroundColor(Color::WHITE),
                    ));
                    parent.spawn((
                        Node{
                            width: Val::Vh(2.),
                            height: Val::Vh(0.5),
                            position_type: PositionType::Absolute,
                            bottom: Val::Vh(0.25),
                            left: Val::Vh(0.25),
                            ..default()
                        },
                        BackgroundColor(Color::WHITE),
                    ));
                });
        });
}

fn spawn_player(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
){
    let far_distance = (RENDER_DISTANCE*CHUNK_SIZE as i32) as f32;
    let drone_controller = DroneController::new();

    /*commands.spawn((
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
        Player,
        drone_controller.clone(),

        RigidBody::Dynamic,
        Collider::cuboid(drone_controller.size.x, drone_controller.size.y, drone_controller.size.z)
        ));*/

    /*let mut render_image = Image::new_fill(
        Extent3d{
            width: 720,
            height: 600,
            ..default()
        },
        TextureDimension::D2,
        &[0,0,0,255],
        TextureFormat::Bgra8UnormSrgb,
        RenderAssetUsages::default(),
    );
    render_image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT;


    let render_image_handle = images.add(render_image);*/

    let drone = commands.spawn((
        Player,
        drone_controller.clone(),
        Transform::from_xyz(1400., -50., 750.).with_rotation(Quat::from_rotation_y(45f32.to_radians())),
        RigidBody::Dynamic,
        Collider::cuboid(drone_controller.size.x, drone_controller.size.y, drone_controller.size.z),
    ));

    let camera = commands.spawn((
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
        Projection::from(PerspectiveProjection{
            fov: 45f32.to_radians(),
            ..default()
        }),
        DroneCamera,
        PostProcessSettings{
            noise_strenght: 0.1,
            ..default()
        }
    ));

}

#[derive(Component)]
struct DroneCamera;

#[derive(Component, Clone)]
struct DroneController {
    size: Vec3,
    
    motor_strength: f32,
    angular_speed: f32,

    top_speed: f32,
    cam_offset: Transform,
    
    drag_side: f32,
    drag_top: f32,

    roll_pid: PID, // Now directly holds the PID controller
    pitch_pid: PID, // Now directly holds the PID controller
    yaw_pid: PID, // Now directly holds the PID controller
    impulse_threshold: f32, // When to apply impulse correction
    impulse_gain: f32, // Strength of impulse correction
}

impl DroneController {
    fn new() -> Self {
        Self {
            size: Vec3::new(0.17, 0.05, 0.17),
            motor_strength: 10.0,
            angular_speed: 6.0,

            top_speed: 50.0,
            cam_offset: Transform::from_xyz(0.0,0.09,0.00).with_rotation(Quat::from_rotation_x(25f32.to_radians())),
            drag_top: 0.994,
            drag_side: 0.998,

            roll_pid: PID::new(1., 0.0, 0.12), // Default gains
            pitch_pid: PID::new(1., 0.0, 0.12), // Default gains
            yaw_pid: PID::new(1., 0.0, 0.12), // Default gains
            impulse_threshold: 0.1, // Threshold for impulse correction
            impulse_gain: 0.2, // Strength of impulse correction
        }
    }
}

#[derive(Clone)]
struct PID {
    kp: f32,
    ki: f32,
    kd: f32,
    prev_error: f32,
    integral: f32,
}

impl PID {
    fn new(kp: f32, ki: f32, kd: f32) -> Self {
        Self {
            kp,
            ki,
            kd,
            prev_error: 0.0,
            integral: 0.0,
        }
    }

    fn update(&mut self, error: f32, delta_time: f32) -> f32 {
        if delta_time <= 0.0001 {
            return 0.0; // Prevents division by zero
        }

        // Prevent integral windup
        self.integral += error * delta_time;
        self.integral = self.integral.clamp(-5.0, 5.0); 

        // Compute derivative with smoothing
        let smoothing = 0.1; 
        let derivative = ((error - self.prev_error) / delta_time).lerp(self.prev_error, smoothing);

        self.prev_error = error;

        // Compute PID output
        let output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative);

        output.clamp(-5.0, 5.0) // Prevent extreme forces
    }
}


fn apply_impulse_correction(
    angular_velocity: Vec3,
    target_angular_velocity: Vec3,
    impulse_threshold: f32,
    impulse_gain: f32,
) -> Vec3 {
    let error = target_angular_velocity - angular_velocity;

    if error.length() > impulse_threshold {
        return error * impulse_gain; // Apply impulse correction
    }

    Vec3::ZERO
}

fn drone_controller(
    time: Res<Time>,
    mut drone: Query<(&mut LinearVelocity, &mut AngularVelocity, &Transform, &mut DroneController)>,
    gamepads: Query<(Entity, &Gamepad)>,
){
    let dt = time.delta_secs()*60.;
    let (mut linear, mut angular, transform, mut controller) = drone.get_single_mut().unwrap();

    for (entity, gamepad) in &gamepads{
        let throtle = gamepad.get(GamepadAxis::LeftStickY).unwrap().max(0.);
        
        let mut yaw = gamepad.get(GamepadAxis::LeftStickX).unwrap();
        let mut pitch = gamepad.get(GamepadAxis::RightStickY).unwrap();
        let mut roll = gamepad.get(GamepadAxis::RightStickX).unwrap();
        let curve = 1.5;

        yaw = adjust_input(yaw, curve);
        pitch = adjust_input(pitch, curve);
        roll = adjust_input(roll, curve);

        let target_ang_vel = -Vec3::new(pitch, yaw, roll) *controller.angular_speed;
        let local_ang_vel = transform.rotation.inverse() * angular.0;

        let pid_torque = Vec3::new(
            controller.pitch_pid.update(target_ang_vel.x -  local_ang_vel.x, dt),
            controller.yaw_pid.update(target_ang_vel.y -  local_ang_vel.y, dt),
            controller.roll_pid.update(target_ang_vel.z -  local_ang_vel.z, dt),
        );
        //let pid_torque = target_ang_vel;
        
        //println!("{}", pid_torque);

        let impulse = apply_impulse_correction(
            local_ang_vel,
            target_ang_vel,
            controller.impulse_threshold,
            controller.impulse_gain
        );
        angular.0 += transform.rotation * (pid_torque+impulse);
        

        let mut local_lin = transform.rotation.inverse() * linear.0;
        let top_speed = Vec3::Y * controller.top_speed;
        let relative_wind = (controller.top_speed - local_lin.y).clamp(0., controller.top_speed);
        let throttle_effectiveness = relative_wind / controller.top_speed;

        let drag_dot = Vec3::Y.dot(local_lin.normalize_or(Vec3::Y)) /2. +0.5;
        let drag_coeficient = controller.drag_side.lerp(controller.drag_top, drag_dot);

        local_lin *= drag_coeficient;

        local_lin += Vec3::Y * throtle * dt * throttle_effectiveness;

        linear.0 = transform.rotation * local_lin;





        //println!("{}, {}, {}, {}",throtle, movedir, speed, airspeed)
    }
}


fn adjust_input(x: f32, exponent: f32) -> f32 {
    let sign = x.signum(); // Keep track of the original sign
    let abs_x = x.abs(); // Work with positive values
    sign * abs_x.powf(exponent) // Apply non-linear scaling and restore sign
}

fn move_drone_camera(
    mut drone_cams: Query<(Entity, &mut Transform), With<Camera3d>>,
    drones: Query<(Entity, &Transform, &DroneController), Without<Camera3d>>
){
    let (cam, mut cam_transform) = drone_cams.get_single_mut().unwrap();
    let (drone, drone_transform, controller) = drones.get_single().unwrap();

    cam_transform.translation = drone_transform.translation+controller.cam_offset.translation;
    cam_transform.rotation = drone_transform.rotation * controller.cam_offset.rotation;
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

fn handle_nearby_colliders(
    mut commands: Commands,
    chunks: Query<(Entity, &Chunk, Has<Collider>), Without<ProcessingLock>>,
    player: Query<&Transform, With<Player>>
){
    let player = player.get_single().unwrap();
    let player_chunkpos = ChunkPosition::from_world_pos(player.translation.as_ivec3()).as_ivec3();
    for (entity, chunk, has_collider) in chunks.iter(){
        let dist = (player_chunkpos - chunk.chunk_position.as_ivec3()).abs();
        if dist.x <= 1 && dist.y <= 1 && dist.z <= 1 {
            if !has_collider {
                commands.entity(entity).insert((NeedsCollider, ProcessingLock));
            }
        }
        else{
            commands.entity(entity).remove::<Collider>();
            commands.entity(entity).remove::<RigidBody>();
        }
    }  
}

#[derive(Resource)]
struct PreloadedAssets {
    material_handle: Handle<StandardMaterial>,
    // Add other asset handles as needed
}

// System to preload assets
fn preload_assets(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    // Add other asset resources as needed
) {
    // Create and add the material to the asset server
    let material_handle = materials.add(StandardMaterial {
        base_color: Color::srgb(0.8, 0.9, 0.8),
        cull_mode: None,
        ..Default::default()
    });

    // Insert the handles into the PreloadedAssets resource
    commands.insert_resource(PreloadedAssets {
        material_handle,
        // Initialize other asset handles
    });
}

// System to spawn queued chunks
fn spawn_queued_chunks(
    mut commands: Commands,
    mut cqueue: ResMut<ChunkLoadQueue>,
    preloaded_assets: Res<PreloadedAssets>,
    mut chunks: ResMut<Chunks>,
) {
    let max_spawns_per_frame = 500;

    // Collect up to `max_spawns_per_frame` positions from the queue
    let positions: Vec<IVec3> = cqueue.queue.drain(..).take(max_spawns_per_frame).collect();

    // Generate chunk data in parallel (excluding asset creation)
    let chunk_data: Vec<(ChunkPosition, Chunk, Transform, Aabb)> = positions
        .par_iter()
        .map(|&pos| {
            let chunk_position = ChunkPosition::from_ivec(pos);
            let chunk = Chunk::new(&chunk_position);
            let transform = Transform::from_translation(
                chunk_position.as_ivec3().as_vec3() * CHUNK_SIZE as f32,
            );
            let aabb = Aabb::from_min_max(Vec3::ZERO, Vec3::splat(CHUNK_SIZE as f32));

            (chunk_position, chunk, transform, aabb)
        })
        .collect();

    // Spawn entities sequentially using preloaded asset handles
    for (chunk_position, chunk, transform, aabb) in chunk_data {
        let chunk_ent = commands
            .spawn((
                chunk,
                transform,
                NeedsGeneration,
                MeshMaterial3d(preloaded_assets.material_handle.clone()),
                aabb,
                ProcessingLock,
            ))
            .id();

        chunks.insert(chunk_position, chunk_ent);
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
    commands: ParallelCommands,
    chunk_ents: Query<(Entity, &Chunk), With<NeedsGeneration>>,
) {
    let thread_pool = AsyncComputeTaskPool::get();

    chunk_ents.par_iter().for_each(|(entity, chunk)|{
        let chunk_pos = chunk.chunk_position.clone(); // Clone the chunk data
        let task = thread_pool.spawn(async move {
            let start_time = Instant::now(); // Record the start time
            let data = Chunk::generate(chunk_pos);
            let duration = start_time.elapsed(); // Measure elapsed time
            //println!("Generation completed in: {:?}", duration);
            data
        });
        commands.command_scope(|mut comm|{
            comm.entity(entity).insert((ProcessingGeneration(task), ProcessingLock));
            comm.entity(entity).remove::<NeedsGeneration>();
        });
    });
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
    par_commands: ParallelCommands,
    chunk_ents: Query<(Entity, &Chunk), With<NeedsMeshing>>,
) {
    let thread_pool = AsyncComputeTaskPool::get();

    chunk_ents.par_iter().for_each(|(entity, chunk)|{
        let chunk_clone = chunk.clone(); // Clone the chunk data
        let task = thread_pool.spawn(async move {
            let start_time = Instant::now(); // Record the start time
            let mesh = chunk_clone.cull_mesher();
            let duration = start_time.elapsed(); // Measure elapsed time
            let collider = chunk_clone.construct_collider();
            println!("Meshing completed in: {:?}", duration);
            mesh
        });
        par_commands.command_scope(|mut commands|{
            commands.entity(entity).insert(ProcessingMeshing(task));
            commands.entity(entity).remove::<NeedsMeshing>();
        });
    });
}

fn poll_chunk_meshing(
    mut commands: Commands,
    mut chunk_ents: Query<(Entity, &mut ProcessingMeshing)>,
    mut meshes: ResMut<Assets<Mesh>>,
){
    for (entity, mut gen_task) in chunk_ents.iter_mut(){
        if let Some(mesh) = block_on(future::poll_once(&mut gen_task.0)){
            commands.entity(entity).remove::<ProcessingMeshing>();
            commands.entity(entity).remove::<ProcessingLock>();
            let mesh_handle = meshes.add(mesh);

            commands.entity(entity).insert(Mesh3d(mesh_handle));
            //println!("meshed")
        }
    }
}

fn construct_colliders_async(
    par_commands: ParallelCommands,
    chunk_ents: Query<(Entity, &Chunk), With<NeedsCollider>>
){
    let thread_pool = AsyncComputeTaskPool::get();
    chunk_ents.par_iter().for_each(|(entity, chunk)|{
        let chunk_clone = chunk.clone();
        let task = thread_pool.spawn( async move{
            let collider = chunk_clone.construct_collider();
            collider
        });
        par_commands.command_scope(|mut commands|{
            commands.entity(entity).insert(ProcessingCollider(task));
            commands.entity(entity).remove::<NeedsCollider>();
        })
    })
}

fn poll_colliders(
    mut commands: Commands,
    mut chunk_ents: Query<(Entity, &mut ProcessingCollider)>
){
    for (ent, mut coll_task) in chunk_ents.iter_mut(){
        if let Some(coll) = block_on(future::poll_once(&mut coll_task.0)){
            if let Some(collider) = coll{
                commands.entity(ent).insert((collider, RigidBody::Static));
                println!("spawned collider")
            }
            commands.entity(ent).remove::<ProcessingCollider>();
            commands.entity(ent).remove::<ProcessingLock>();
        }
    }
}


#[derive(Component)]
struct ProcessingLock;

