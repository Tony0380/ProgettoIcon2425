import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.transport_data import CityGraph
from ml_models.ml_pathfinder_integration import MLEnhancedPathfinder
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=== ML-ENHANCED PATHFINDER QUICK TEST ===")
    
    # Setup
    city_graph = CityGraph()
    ml_pathfinder = MLEnhancedPathfinder(city_graph)
    
    # Training veloce
    print("\n[1] Training ML models (small dataset)...")
    training_results = ml_pathfinder.train_ml_models(n_scenarios=300)
    
    # Test singolo
    print("\n[2] Testing ML-enhanced route finding...")
    
    origin, destination = 'Milano', 'Roma'
    user_profile = {
        'user_age': 32,
        'user_income': 45000,
        'price_sensitivity': 0.6,
        'time_priority': 0.7,
        'comfort_priority': 0.5
    }
    
    # ML-Enhanced route
    ml_route, metadata = ml_pathfinder.find_ml_enhanced_route(
        origin, destination, 
        user_profile=user_profile,
        season='summer'
    )
    
    # Base A* route per confronto
    base_route = ml_pathfinder.base_pathfinder.multi_objective_astar(origin, destination)
    
    print(f"\n=== RESULTS COMPARISON: {origin} -> {destination} ===")
    
    if base_route:
        print(f"\nBASE A* ROUTE:")
        print(f"   Path: {' -> '.join(base_route.path)}")
        print(f"   Cost: €{base_route.total_cost:.2f}")
        print(f"   Time: {base_route.total_time:.1f}h")
        print(f"   Score: {base_route.normalized_score:.3f}")
    
    if ml_route:
        print(f"\nML-ENHANCED ROUTE:")
        print(f"   Path: {' -> '.join(ml_route.path)}")
        print(f"   Cost: €{ml_route.total_cost:.2f}")
        print(f"   Time: {ml_route.total_time:.1f}h")
        print(f"   Score: {ml_route.normalized_score:.3f}")
        print(f"   User Profile: {metadata['user_profile']}")
        print(f"   Weights: {metadata['personalized_weights']}")
    
    # Improvements
    if base_route and ml_route:
        cost_improvement = ((base_route.total_cost - ml_route.total_cost) / base_route.total_cost) * 100
        time_improvement = ((base_route.total_time - ml_route.total_time) / base_route.total_time) * 100
        score_improvement = ((ml_route.normalized_score - base_route.normalized_score) / base_route.normalized_score) * 100
        
        print(f"\n=== IMPROVEMENTS ===")
        print(f"Cost: {cost_improvement:+.1f}%")
        print(f"Time: {time_improvement:+.1f}%") 
        print(f"Score: {score_improvement:+.1f}%")
    
    print(f"\n=== TRAINING SUMMARY ===")
    print(f"Price Predictor: {ml_pathfinder.price_predictor.best_model_name} (R² = {training_results['price_results'][ml_pathfinder.price_predictor.best_model_name]['test_r2']:.3f})")
    print(f"User Classifier: {ml_pathfinder.user_classifier.best_model_name} (Acc = {training_results['user_results'][ml_pathfinder.user_classifier.best_model_name]['test_accuracy']:.3f})")
    print(f"Time Estimator: R² = {training_results['time_results']['cv_r2_mean']:.3f} ± {training_results['time_results']['cv_r2_std']:.3f}")
    
    print(f"\n[DONE] ML-Enhanced Pathfinder test completed successfully!")

if __name__ == "__main__":
    main()