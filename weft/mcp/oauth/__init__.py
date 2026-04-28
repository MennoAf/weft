"""Vestigial namespace from the prior Weft-as-OAuth-server architecture.

Weft no longer hosts OAuth endpoints. Supabase's OAuth Server is the
authorization server (see ``weft/mcp/oauth_consent.py`` for the consent
page Weft still serves). This package is kept as an empty namespace so
external imports of ``weft.mcp.oauth`` don't ImportError mid-rollback.
"""
