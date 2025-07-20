from langgraph.graph import StateGraph
from langgraph_nodes import (
    state_manager_node,
    waypoint_planner_node,
    action_planner_node
)

# === 1. Initialize LangGraph ===
graph = StateGraph()

# === 2. Add Agent Nodes ===
graph.add_node("SensorUpdate", state_manager_node)
graph.add_node("WaypointPlan", waypoint_planner_node)
graph.add_node("ActionPlan", action_planner_node)

# === 3. Set Node Execution Order ===
graph.set_entry_point("SensorUpdate")
graph.add_edge("SensorUpdate", "WaypointPlan")
graph.add_edge("WaypointPlan", "ActionPlan")

# === 4. Compile the LangGraph ===
compiled_graph = graph.compile()