#!/usr/bin/env python3

import numpy as np
import genesis as gs

def main():
    gs.init()
    
    # Create output file
    with open("vertex_state_data.txt", "w") as f:
        f.write("Genesis Vertex State Data Comparison\n")
        f.write("===================================\n\n")
        
        print("=== Testing Two Fixed Cubes ===")
        f.write("=== Two Fixed Cubes ===\n")
        
        # Create scene with two fixed cubes
        scene1 = gs.Scene()
        cube1 = scene1.add_entity(gs.morphs.Box(size=(0.5, 0.5, 1.0), pos=(1.0, 1.0, 0.5), fixed=True))
        cube2 = scene1.add_entity(gs.morphs.Box(size=(0.5, 0.5, 1.0), pos=(-1.0, 1.0, 0.5), fixed=True))
        
        scene1.build()
        scene1.step()
        
        solver1 = scene1.sim.rigid_solver
        
        # Update vertex positions
        import genesis.engine.solvers.rigid.rigid_solver_decomp as rigid_solver_decomp
        rigid_solver_decomp.kernel_update_all_verts(
            geoms_state=solver1.geoms_state,
            verts_info=solver1.verts_info,
            free_verts_state=solver1.free_verts_state,
            fixed_verts_state=solver1.fixed_verts_state,
        )
        
        # Get vertex state data
        fixed_verts_pos = solver1.fixed_verts_state.pos.to_numpy()
        free_verts_pos = solver1.free_verts_state.pos.to_numpy()
        
        f.write(f"Fixed vertices count: {len(fixed_verts_pos)}\n")
        f.write(f"Free vertices count: {free_verts_pos.shape[0]}\n")
        f.write(f"Fixed vertex positions:\n")
        for i, pos in enumerate(fixed_verts_pos):
            f.write(f"  {i}: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]\n")
        
        f.write(f"Free vertex positions (batch 0):\n")
        free_batch0 = free_verts_pos[:, 0, :]
        for i, pos in enumerate(free_batch0):
            f.write(f"  {i}: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]\n")
        
        print(f"Fixed cubes: {len(fixed_verts_pos)} fixed verts, {free_verts_pos.shape[0]} free verts")
        
        # Clean up
        del scene1, solver1
        
        print("\n=== Testing Two Free Cubes ===")
        f.write(f"\n=== Two Free Cubes ===\n")
        
        # Create scene with two free cubes
        scene2 = gs.Scene()
        cube3 = scene2.add_entity(gs.morphs.Box(size=(0.5, 0.5, 1.0), pos=(1.0, 1.0, 0.5), fixed=False))
        cube4 = scene2.add_entity(gs.morphs.Box(size=(0.5, 0.5, 1.0), pos=(-1.0, 1.0, 0.5), fixed=False))
        
        scene2.build()
        scene2.step()
        
        solver2 = scene2.sim.rigid_solver
        
        # Update vertex positions
        rigid_solver_decomp.kernel_update_all_verts(
            geoms_state=solver2.geoms_state,
            verts_info=solver2.verts_info,
            free_verts_state=solver2.free_verts_state,
            fixed_verts_state=solver2.fixed_verts_state,
        )
        
        # Get vertex state data
        fixed_verts_pos2 = solver2.fixed_verts_state.pos.to_numpy()
        free_verts_pos2 = solver2.free_verts_state.pos.to_numpy()
        
        f.write(f"Fixed vertices count: {len(fixed_verts_pos2)}\n")
        f.write(f"Free vertices count: {free_verts_pos2.shape[0]}\n")
        f.write(f"Fixed vertex positions:\n")
        for i, pos in enumerate(fixed_verts_pos2):
            f.write(f"  {i}: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]\n")
        
        f.write(f"Free vertex positions (batch 0):\n")
        free_batch0_2 = free_verts_pos2[:, 0, :]
        for i, pos in enumerate(free_batch0_2):
            f.write(f"  {i}: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]\n")
        
        print(f"Free cubes: {len(fixed_verts_pos2)} fixed verts, {free_verts_pos2.shape[0]} free verts")
    
    print("\nData saved to vertex_state_data.txt")

if __name__ == "__main__":
    main()
