"""Counterparty risk modeling for venue failures and operational risks."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import numpy as np


class VenueStatus(Enum):
    """Venue operational status."""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"  # Partial functionality
    HALTED = "halted"  # Trading halted, positions locked
    FAILED = "failed"  # Complete failure, haircuts applied


@dataclass
class CounterpartyEvent:
    """Counterparty/venue failure event."""
    
    venue: str
    event_type: VenueStatus
    haircut: float = 0.0  # Loss on assets (0.0 = no loss, 1.0 = total loss)
    recovery_time_hours: float = 0.0  # Time until operational
    contagion_probability: float = 0.0  # Probability of cascade to other venues
    
    def apply_haircut(self, asset_value: float) -> float:
        """Apply haircut to asset value."""
        return asset_value * (1.0 - self.haircut)


@dataclass
class CounterpartyScenario:
    """Counterparty stress scenario."""
    
    name: str
    description: str
    events: List[CounterpartyEvent]
    probability: float  # Annual probability
    
    def total_haircut(self, venue_exposures: Dict[str, float]) -> float:
        """Calculate total loss from scenario given venue exposures."""
        total_loss = 0.0
        
        for event in self.events:
            if event.venue in venue_exposures:
                exposure = venue_exposures[event.venue]
                loss = exposure * event.haircut
                total_loss += loss
        
        return total_loss


class CounterpartyRiskModel:
    """
    Counterparty risk model for venue failures.
    
    Models:
    - Single venue failure with haircuts
    - Cascading failures (contagion)
    - Recovery scenarios
    - Operational risk (halts, degradation)
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize counterparty risk model."""
        self.rng = np.random.default_rng(seed)
        self.predefined_scenarios = self._create_predefined_scenarios()
    
    def _create_predefined_scenarios(self) -> Dict[str, CounterpartyScenario]:
        """Create library of predefined counterparty scenarios."""
        return {
            "single_venue_minor": CounterpartyScenario(
                name="Single Venue Minor Failure",
                description="One venue experiences temporary halt with 5% haircut",
                events=[
                    CounterpartyEvent(
                        venue="default",
                        event_type=VenueStatus.HALTED,
                        haircut=0.05,
                        recovery_time_hours=24.0,
                        contagion_probability=0.1
                    )
                ],
                probability=0.10  # 10% annual
            ),
            "single_venue_major": CounterpartyScenario(
                name="Single Venue Major Failure",
                description="One venue fails with 30% haircut (FTX-style)",
                events=[
                    CounterpartyEvent(
                        venue="default",
                        event_type=VenueStatus.FAILED,
                        haircut=0.30,
                        recovery_time_hours=float('inf'),
                        contagion_probability=0.25
                    )
                ],
                probability=0.02  # 2% annual
            ),
            "cascade_failure": CounterpartyScenario(
                name="Cascading Venue Failures",
                description="Primary venue fails, triggers secondary failures",
                events=[
                    CounterpartyEvent(
                        venue="primary",
                        event_type=VenueStatus.FAILED,
                        haircut=0.40,
                        contagion_probability=0.5
                    ),
                    CounterpartyEvent(
                        venue="secondary",
                        event_type=VenueStatus.DEGRADED,
                        haircut=0.15,
                        contagion_probability=0.2
                    )
                ],
                probability=0.005  # 0.5% annual
            ),
            "market_halt": CounterpartyScenario(
                name="Market-Wide Trading Halt",
                description="All venues halt trading temporarily (no haircut)",
                events=[
                    CounterpartyEvent(
                        venue="binance",
                        event_type=VenueStatus.HALTED,
                        haircut=0.0,
                        recovery_time_hours=6.0
                    ),
                    CounterpartyEvent(
                        venue="dydx",
                        event_type=VenueStatus.HALTED,
                        haircut=0.0,
                        recovery_time_hours=6.0
                    )
                ],
                probability=0.05  # 5% annual
            ),
            "partial_segregation": CounterpartyScenario(
                name="Partial Fund Segregation Failure",
                description="10% of assets not properly segregated, lost in failure",
                events=[
                    CounterpartyEvent(
                        venue="default",
                        event_type=VenueStatus.FAILED,
                        haircut=0.10,
                        recovery_time_hours=float('inf')
                    )
                ],
                probability=0.03  # 3% annual
            )
        }
    
    def get_scenario(self, scenario_name: str) -> CounterpartyScenario:
        """Get predefined scenario by name."""
        if scenario_name not in self.predefined_scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        return self.predefined_scenarios[scenario_name]
    
    def create_custom_scenario(
        self,
        name: str,
        venue_haircuts: Dict[str, float],
        event_type: VenueStatus = VenueStatus.FAILED,
        probability: float = 0.01
    ) -> CounterpartyScenario:
        """
        Create custom counterparty scenario.
        
        Args:
            name: Scenario name
            venue_haircuts: Dictionary of venue -> haircut (0.0 to 1.0)
            event_type: Type of event
            probability: Annual probability
        
        Returns:
            CounterpartyScenario
        """
        events = [
            CounterpartyEvent(
                venue=venue,
                event_type=event_type,
                haircut=haircut,
                recovery_time_hours=float('inf') if event_type == VenueStatus.FAILED else 24.0
            )
            for venue, haircut in venue_haircuts.items()
        ]
        
        return CounterpartyScenario(
            name=name,
            description=f"Custom scenario: {name}",
            events=events,
            probability=probability
        )
    
    def simulate_contagion(
        self,
        initial_event: CounterpartyEvent,
        venue_network: Dict[str, List[str]],
        venue_exposures: Dict[str, float]
    ) -> List[CounterpartyEvent]:
        """
        Simulate contagion from initial failure.
        
        Args:
            initial_event: Initial failure event
            venue_network: Graph of venue connections
            venue_exposures: Exposure at each venue
        
        Returns:
            List of all events including cascades
        """
        events = [initial_event]
        affected_venues = {initial_event.venue}
        
        # Simple contagion: each failure can trigger connected venues
        to_check = [initial_event.venue]
        
        while to_check:
            venue = to_check.pop(0)
            
            # Get connected venues
            connected = venue_network.get(venue, [])
            
            for connected_venue in connected:
                if connected_venue in affected_venues:
                    continue
                
                # Check if contagion occurs
                base_prob = initial_event.contagion_probability
                
                # Increase probability based on exposure size
                exposure_factor = venue_exposures.get(connected_venue, 0) / 1_000_000  # Normalize
                contagion_prob = min(base_prob * (1 + exposure_factor), 0.9)
                
                if self.rng.random() < contagion_prob:
                    # Contagion occurs, but typically with lower haircut
                    secondary_haircut = initial_event.haircut * 0.5
                    
                    secondary_event = CounterpartyEvent(
                        venue=connected_venue,
                        event_type=VenueStatus.DEGRADED,
                        haircut=secondary_haircut,
                        recovery_time_hours=48.0,
                        contagion_probability=contagion_prob * 0.5
                    )
                    
                    events.append(secondary_event)
                    affected_venues.add(connected_venue)
                    to_check.append(connected_venue)
        
        return events
    
    def calculate_expected_loss(
        self,
        venue_exposures: Dict[str, float],
        time_horizon_years: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate expected loss from counterparty risk.
        
        Args:
            venue_exposures: Exposure at each venue
            time_horizon_years: Time horizon
        
        Returns:
            Dictionary with loss metrics
        """
        total_exposure = sum(venue_exposures.values())
        expected_loss = 0.0
        
        for scenario in self.predefined_scenarios.values():
            # Adjust scenario for specific venues
            scenario_loss = 0.0
            
            for event in scenario.events:
                # Apply to each venue if "default", or specific venue
                if event.venue == "default":
                    # Apply to all venues
                    for venue, exposure in venue_exposures.items():
                        scenario_loss += exposure * event.haircut
                elif event.venue in venue_exposures:
                    scenario_loss += venue_exposures[event.venue] * event.haircut
            
            # Expected loss = probability * loss
            annual_probability = scenario.probability
            horizon_probability = 1 - (1 - annual_probability) ** time_horizon_years
            expected_loss += horizon_probability * scenario_loss
        
        return {
            "expected_loss": expected_loss,
            "expected_loss_pct": expected_loss / total_exposure if total_exposure > 0 else 0.0,
            "total_exposure": total_exposure,
            "time_horizon_years": time_horizon_years
        }
    
    def venue_concentration_risk(
        self,
        venue_exposures: Dict[str, float],
        concentration_limit: float = 0.3
    ) -> Dict[str, any]:
        """
        Assess venue concentration risk.
        
        Args:
            venue_exposures: Exposure at each venue
            concentration_limit: Maximum acceptable concentration (e.g., 0.3 = 30%)
        
        Returns:
            Concentration risk metrics
        """
        total_exposure = sum(venue_exposures.values())
        
        if total_exposure == 0:
            return {
                "max_concentration": 0.0,
                "max_concentration_venue": None,
                "exceeds_limit": False,
                "herfindahl_index": 0.0
            }
        
        concentrations = {
            venue: exposure / total_exposure
            for venue, exposure in venue_exposures.items()
        }
        
        max_venue = max(concentrations.items(), key=lambda x: x[1])
        
        # Herfindahl-Hirschman Index (0 = perfect diversification, 1 = concentrated)
        hhi = sum(c**2 for c in concentrations.values())
        
        return {
            "max_concentration": max_venue[1],
            "max_concentration_venue": max_venue[0],
            "exceeds_limit": max_venue[1] > concentration_limit,
            "herfindahl_index": hhi,
            "concentrations": concentrations
        }
