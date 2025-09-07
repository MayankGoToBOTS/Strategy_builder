# backend/tests/test_strategy_builder.py
import pytest
from httpx import AsyncClient
from app.main import app
from app.core.schemas.strategy_spec import StrategyRequest
from app.deps.mongo_client import MongoManager
from app.deps.redis_client import RedisManager

class TestStrategyBuilder:
    """Test strategy builder orchestrator"""
    
    async def setup_databases(self):
        """Initialize databases for testing"""
        try:
            await MongoManager.initialize()
            await RedisManager.initialize()
        except Exception:
            pass  # Ignore if already initialized
    
    async def cleanup_databases(self):
        """Cleanup databases after testing"""
        try:
            await MongoManager.close()
            await RedisManager.close()
        except Exception:
            pass
    
    @pytest.mark.asyncio
    async def test_build_strategy_endpoint(self, sample_strategy_request):
        """Test strategy building endpoint"""
        await self.setup_databases()
        
        try:
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/strategy/build",
                    json=sample_strategy_request.model_dump()
                )
                
                assert response.status_code == 200
                data = response.json()
                
                assert "human_report" in data
                assert "strategy_spec" in data
                assert len(data["human_report"]) > 100  # Should have substantial content
                assert "id" in data["strategy_spec"]
        finally:
            await self.cleanup_databases()
    
    @pytest.mark.asyncio
    async def test_get_strategy_templates(self):
        """Test getting strategy templates"""
        await self.setup_databases()
        
        try:
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/api/v1/strategy/templates")
                
                assert response.status_code == 200
                data = response.json()
                
                assert "templates" in data
                templates = data["templates"]
                
                # Should have all 5 strategy types
                expected_types = ["scalping", "grid", "dca", "momentum", "pattern_rule"]
                for strategy_type in expected_types:
                    assert strategy_type in templates
                    template = templates[strategy_type]
                    assert "name" in template
                    assert "description" in template
                    assert "default_params" in template
        finally:
            await self.cleanup_databases()
    
    @pytest.mark.asyncio
    async def test_list_strategies_empty(self):
        """Test listing strategies when none exist"""
        await self.setup_databases()
        
        try:
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/api/v1/strategy/")
                
                assert response.status_code == 200
                data = response.json()
                
                assert "strategies" in data
                assert "total" in data
                assert data["total"] == 0
                assert len(data["strategies"]) == 0
        finally:
            await self.cleanup_databases()
    
    @pytest.mark.asyncio
    async def test_build_and_retrieve_strategy(self, sample_strategy_request):
        """Test building a strategy and then retrieving it"""
        await self.setup_databases()
        
        try:
            async with AsyncClient(app=app, base_url="http://test") as client:
                # Build strategy
                build_response = await client.post(
                    "/api/v1/strategy/build",
                    json=sample_strategy_request.model_dump()
                )
                
                assert build_response.status_code == 200
                build_data = build_response.json()
                strategy_id = build_data["strategy_spec"]["id"]
                
                # Retrieve strategy
                get_response = await client.get(f"/api/v1/strategy/{strategy_id}")
                
                assert get_response.status_code == 200
                strategy_data = get_response.json()
                
                assert strategy_data["id"] == strategy_id
                assert "bots" in strategy_data
                assert "portfolio" in strategy_data
        finally:
            await self.cleanup_databases()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])









# # backend/tests/test_strategy_builder.py
# import pytest
# from httpx import AsyncClient
# from app.main import app
# from app.core.schemas.strategy_spec import StrategyRequest

# class TestStrategyBuilder:
#     """Test strategy builder orchestrator"""
    
#     @pytest.mark.asyncio
#     async def test_build_strategy_endpoint(self, sample_strategy_request):
#         """Test strategy building endpoint"""
#         async with AsyncClient(app=app, base_url="http://test") as client:
#             response = await client.post(
#                 "/api/v1/strategy/build",
#                 json=sample_strategy_request.dict()
#             )
            
#             assert response.status_code == 200
#             data = response.json()
            
#             assert "human_report" in data
#             assert "strategy_spec" in data
#             assert len(data["human_report"]) > 100  # Should have substantial content
#             assert "id" in data["strategy_spec"]
    
#     @pytest.mark.asyncio
#     async def test_get_strategy_templates(self):
#         """Test getting strategy templates"""
#         async with AsyncClient(app=app, base_url="http://test") as client:
#             response = await client.get("/api/v1/strategy/templates")
            
#             assert response.status_code == 200
#             data = response.json()
            
#             assert "templates" in data
#             templates = data["templates"]
            
#             # Should have all 5 strategy types
#             expected_types = ["scalping", "grid", "dca", "momentum", "pattern_rule"]
#             for strategy_type in expected_types:
#                 assert strategy_type in templates
#                 template = templates[strategy_type]
#                 assert "name" in template
#                 assert "description" in template
#                 assert "default_params" in template
    
#     @pytest.mark.asyncio
#     async def test_list_strategies_empty(self):
#         """Test listing strategies when none exist"""
#         async with AsyncClient(app=app, base_url="http://test") as client:
#             response = await client.get("/api/v1/strategy/")
            
#             assert response.status_code == 200
#             data = response.json()
            
#             assert "strategies" in data
#             assert "total" in data
#             assert data["total"] == 0
#             assert len(data["strategies"]) == 0
    
#     @pytest.mark.asyncio
#     async def test_build_and_retrieve_strategy(self, sample_strategy_request):
#         """Test building a strategy and then retrieving it"""
#         async with AsyncClient(app=app, base_url="http://test") as client:
#             # Build strategy
#             build_response = await client.post(
#                 "/api/v1/strategy/build",
#                 json=sample_strategy_request.dict()
#             )
            
#             assert build_response.status_code == 200
#             build_data = build_response.json()
#             strategy_id = build_data["strategy_spec"]["id"]
            
#             # Retrieve strategy
#             get_response = await client.get(f"/api/v1/strategy/{strategy_id}")
            
#             assert get_response.status_code == 200
#             strategy_data = get_response.json()
            
#             assert strategy_data["id"] == strategy_id
#             assert "bots" in strategy_data
#             assert "portfolio" in strategy_data

# if __name__ == "__main__":
#     pytest.main([__file__, "-v"])










# # backend/tests/test_strategy_builder.py
# import pytest
# from httpx import AsyncClient
# from app.core.schemas.strategy_spec import StrategyRequest

# class TestStrategyBuilder:
#     """Test strategy builder orchestrator"""
    
#     @pytest.mark.asyncio
#     async def test_build_strategy_endpoint(self, client: AsyncClient, sample_strategy_request):
#         """Test strategy building endpoint"""
#         response = await client.post(
#             "/api/v1/strategy/build",
#             json=sample_strategy_request.dict()
#         )
        
#         assert response.status_code == 200
#         data = response.json()
        
#         assert "human_report" in data
#         assert "strategy_spec" in data
#         assert len(data["human_report"]) > 100  # Should have substantial content
#         assert "id" in data["strategy_spec"]
    
#     @pytest.mark.asyncio
#     async def test_get_strategy_templates(self, client: AsyncClient):
#         """Test getting strategy templates"""
#         response = await client.get("/api/v1/strategy/templates")
        
#         assert response.status_code == 200
#         data = response.json()
        
#         assert "templates" in data
#         templates = data["templates"]
        
#         # Should have all 5 strategy types
#         expected_types = ["scalping", "grid", "dca", "momentum", "pattern_rule"]
#         for strategy_type in expected_types:
#             assert strategy_type in templates
#             template = templates[strategy_type]
#             assert "name" in template
#             assert "description" in template
#             assert "default_params" in template
    
#     @pytest.mark.asyncio
#     async def test_list_strategies_empty(self, client: AsyncClient):
#         """Test listing strategies when none exist"""
#         response = await client.get("/api/v1/strategy/")
        
#         assert response.status_code == 200
#         data = response.json()
        
#         assert "strategies" in data
#         assert "total" in data
#         assert data["total"] == 0
#         assert len(data["strategies"]) == 0
    
#     @pytest.mark.asyncio
#     async def test_build_and_retrieve_strategy(self, client: AsyncClient, sample_strategy_request):
#         """Test building a strategy and then retrieving it"""
#         # Build strategy
#         build_response = await client.post(
#             "/api/v1/strategy/build",
#             json=sample_strategy_request.dict()
#         )
        
#         assert build_response.status_code == 200
#         build_data = build_response.json()
#         strategy_id = build_data["strategy_spec"]["id"]
        
#         # Retrieve strategy
#         get_response = await client.get(f"/api/v1/strategy/{strategy_id}")
        
#         assert get_response.status_code == 200
#         strategy_data = get_response.json()
        
#         assert strategy_data["id"] == strategy_id
#         assert "bots" in strategy_data
#         assert "portfolio" in strategy_data

# if __name__ == "__main__":
#     pytest.main([__file__, "-v"])