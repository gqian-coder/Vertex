import sys
sys.path.insert(0, 'src')
from data_loader import ExodusDataLoader
import numpy as np

# Load both files
print('Loading data files...')
loader_fine = ExodusDataLoader('dataset/002-Re-148_3-AC-beta-10000-Helios/360-120/cropped.e')
fine_data = loader_fine.load()
loader_fine.close()

loader_interp = ExodusDataLoader('paraview/90-30_to_360-120_interpolated_linear.exo')
interp_data = loader_interp.load()
loader_interp.close()

print('\n' + '='*70)
print('EXTENDED SYSTEMATIC OFFSET SEARCH')
print('='*70)

# Starting point in fine mesh
fine_start = 10
print(f'\nStarting from fine mesh timestep {fine_start}')

# Test offsets from -20 to +20 (extended range)
offset_range = range(-20, 20)
results = []

print(f'\nTesting offsets from {min(offset_range)} to {max(offset_range)}...')
print('(This compares all subsequent timesteps after applying the offset)\n')

for offset in offset_range:
    interp_start = fine_start + offset
    
    # Check if this offset is valid
    if interp_start < 0 or interp_start >= len(interp_data['fields']['time_values']):
        continue
    
    # Calculate how many timesteps we can compare
    max_fine = len(fine_data['fields']['time_values'])
    max_interp = len(interp_data['fields']['time_values'])
    num_steps_to_compare = min(max_fine - fine_start, max_interp - interp_start)
    
    if num_steps_to_compare < 10:  # Need at least 10 steps to compare
        continue
    
    # Calculate cumulative RMSE over all comparable timesteps
    total_rmse = 0
    total_count = 0
    
    for delta in range(num_steps_to_compare):
        fine_idx = fine_start + delta
        interp_idx = interp_start + delta
        
        # Calculate RMSE for all fields at this timestep pair
        rmse_sum = 0
        field_count = 0
        
        for field in ['velocity_0', 'velocity_1', 'pressure', 'temperature']:
            if field in fine_data['fields'] and field in interp_data['fields']:
                fine_vals = fine_data['fields'][field][fine_idx, :]
                interp_vals = interp_data['fields'][field][interp_idx, :]
                rmse = np.sqrt(np.mean((fine_vals - interp_vals)**2))
                rmse_sum += rmse
                field_count += 1
        
        avg_rmse = rmse_sum / field_count if field_count > 0 else float('inf')
        total_rmse += avg_rmse
        total_count += 1
    
    # Average RMSE across all compared timesteps
    avg_total_rmse = total_rmse / total_count if total_count > 0 else float('inf')
    
    results.append({
        'offset': offset,
        'interp_start': interp_start,
        'num_compared': num_steps_to_compare,
        'avg_rmse': avg_total_rmse,
        'total_rmse': total_rmse
    })

# Sort by average RMSE
results.sort(key=lambda x: x['avg_rmse'])

print(f'Top 20 best offsets:')
print(f'  Rank | Offset | Interp Start | Num Compared | Avg RMSE   | Total RMSE')
print(f'  -----|--------|--------------|--------------|------------|------------')

for rank, result in enumerate(results[:20], 1):
    marker = ' ← BEST' if rank == 1 else ''
    print(f'  {rank:4d} |  {result["offset"]:+5d} |     {result["interp_start"]:8d} |    {result["num_compared"]:9d} | {result["avg_rmse"]:.8f} | {result["total_rmse"]:10.4f}{marker}')

# Detailed analysis of best offset
best = results[0]
print(f'\n{"="*70}')
print(f'BEST OFFSET FOUND: {best["offset"]:+d}')
print(f'{"="*70}')
print(f'\nMapping: Fine[t] ←→ Interpolated[t {best["offset"]:+d}]')
print(f'\nThis means:')
print(f'  - Fine mesh step {fine_start} matches Interpolated step {best["interp_start"]}')
print(f'  - Compared {best["num_compared"]} subsequent timestep pairs')
print(f'  - Average RMSE per timestep: {best["avg_rmse"]:.8f}')

# Show full range of offsets tested
print(f'\n\nFull offset scan results (all offsets from {min(offset_range)} to {max(offset_range)}):')
print(f'  Offset | Avg RMSE')
print(f'  -------|----------')
for result in results:
    marker = ' ← BEST' if result['offset'] == best['offset'] else ''
    print(f'  {result["offset"]:+6d} | {result["avg_rmse"]:.8f}{marker}')

print(f'\n{"="*70}')
print(f'CONCLUSION: Use offset = {best["offset"]:+d}')
print(f'Fine mesh step N corresponds to Interpolated step (N {best["offset"]:+d})')
print(f'{"="*70}')
